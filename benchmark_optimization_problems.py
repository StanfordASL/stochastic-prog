import enum
import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)

omega_mean = 1.0 # mean
omega_std = 5.0 # standard deviation

class GaussianMoments(enum.Enum):
    First = omega_mean
    Second = (omega_mean**2 
              + omega_std**2)
    Third = (omega_mean**3 
             + 3 * omega_mean * omega_std**2)
    Fourth = (omega_mean**4 
             + 6 * omega_mean**2 * omega_std**2
             + 3 * omega_std**4)
    Fifth = (omega_mean**5 
             + 10 * omega_mean**3 * omega_std**2
             + 15 * omega_mean * omega_std**4)


class Program:
    def __init__(
            self,
            name: str,
            num_variables: int,
            num_equality_constraints: int,
            num_inequality_constraints: int,
            verbose: bool = False):
        if verbose:
            print("Initializing Program with")
            print("> name          =", name)
            print("> num_variables =", num_variables)
        self._name = name
        self._num_variables = num_variables
        self._num_equality_constraints = num_equality_constraints
        self._num_inequality_constraints = num_inequality_constraints

    @property
    def num_variables(self) -> int:
        return self._num_variables

    @property
    def num_constraints(self) -> int:
        num = (self._num_equality_constraints
               + self._num_inequality_constraints)
        return num

    @property
    def num_equality_constraints(self) -> int:
        return self._num_equality_constraints

    @property
    def num_inequality_constraints(self) -> int:
        return self._num_inequality_constraints

    @property
    def name(self) -> str:
        return self._name

    def initial_guess(self) -> np.array:
        raise NotImplementedError

    def solution(self) -> np.array:
        # unknown since stochastically perturbed program
        raise NotImplementedError

    def objective(self, x: jnp.array) -> float:
        """Returns objective f(x) to minimize."""
        raise NotImplementedError

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        """Returns equality constraints.

        Returns h(x) corresponding to the 
        equality constraints h(x) = 0.

        Args:
            x: optimization variable,
                (_num_variables, ) array

        Returns:
            h_value: value of h(x), 
                (_num_equality_constraints, ) array
        """
        raise NotImplementedError

    def inequality_constraints(
            self, 
            x: jnp.array): 
            # -> tuple[jnp.array, jnp.array, jnp.array]:
        """Returns inequality constraints.

        Returns g(x), g_l, g_u corresponding to the 
        inequality constraints g_l <= g(x) <= g_u.

        Args:
            x: optimization variable,
                (_num_variables, ) array

        Returns:
            g_value: value of h(x), 
                (_num_inequality_constraints, ) array
            g_l: value of g_l, 
                (_num_inequality_constraints, ) array
            g_u: value of g_u, 
                (_num_inequality_constraints, ) array
        """
        raise NotImplementedError

    def test(self):
        x0 = self.initial_guess()
        f0 = self.objective(x0)
        hs = self.equality_constraints(x0)
        gs, _, _ = self.inequality_constraints(x0)
        assert len(x0) == self.num_variables
        assert len(hs) == self.num_equality_constraints
        assert len(gs) == self.num_inequality_constraints
 

class SampledProgram(Program):
    def __init__(
        self,
        deterministic_program: Program,
        sample_size: int = 20,
        add_bias: bool = False):
        p = deterministic_program
        num_equality_constraints = p.num_equality_constraints
        super().__init__(
            name=p.name,
            num_variables=p.num_variables,
            num_equality_constraints=num_equality_constraints,
            num_inequality_constraints=p.num_inequality_constraints)
        self._deterministic_program = p
        self._sample_size = sample_size
        self._add_bias = add_bias
        self._omegas = self.sample_omegas()
        self._omegas_bias = self.sample_omegas_bias()

    def sample_omegas(self) -> np.array:
        omegas = np.random.normal(
            loc=omega_mean, scale=omega_std, 
            size=(self._sample_size, self._num_variables))
        # omegas = np.ones((self._sample_size, self._num_variables))
        return omegas

    def sample_omegas_bias(self) -> np.array:
        omegas_bias = np.random.normal(
            loc=0, scale=2*omega_std, 
            size=(self._sample_size, self._num_equality_constraints))
        return omegas_bias

    def initial_guess(self) -> np.array:
        program = self._deterministic_program
        x0 = program.initial_guess()
        return x0

    def objective(self, x: jnp.array) -> float:
        program = self._deterministic_program
        f_value = program.objective(x)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        def multiply_x_by_omega(
                x: jnp.array, 
                omega: float):
            return x*omega

        xs_perturbed = vmap(multiply_x_by_omega, 
            in_axes=(None, 0))(x, self._omegas)

        program = self._deterministic_program
        hs_value = vmap(
            program.equality_constraints)(xs_perturbed)

        if self._add_bias:
            hs_value = hs_value + self._omegas_bias

        hs_value = jnp.mean(hs_value, axis=0)

        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array):
            # -> tuple[jnp.array, jnp.array, jnp.array]:
        program = self._deterministic_program
        gx, gl, gu = program.inequality_constraints(x)
        return gx, gl, gu


class StochasticTrueAverageValueProgram(Program):
    def __init__(
        self,
        program: Program):
        p = program
        num_equality_constraints = p.num_equality_constraints
        super().__init__(
            name=p.name,
            num_variables=p.num_variables,
            num_equality_constraints=num_equality_constraints,
            num_inequality_constraints=p.num_inequality_constraints)
        self._program = p

    def initial_guess(self) -> np.array:
        program = self._program
        x0 = program.initial_guess()
        return x0

    def objective(self, x: jnp.array) -> float:
        program = self._program
        f_value = program.objective(x)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        program = self._program
        hs_value = program.equality_constraints_average_value(x)
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array):
            # -> tuple[jnp.array, jnp.array, jnp.array]:
        program = self._program
        gx, gl, gu = program.inequality_constraints(x)
        return gx, gl, gu


class Program6(Program):
    def __init__(self):
        super().__init__(
            name="program_6",
            num_variables=2,
            num_equality_constraints=1,
            num_inequality_constraints=2)

    def initial_guess(self) -> np.array:
        x0 = np.array([-1.2, 1.])
        return x0

    def objective(self, x: jnp.array) -> float:
        f_value = (1 - x[0])**2
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2 = x
        hs_value = 10 * jnp.array([
            x2 - x1**2])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        # returns the average value of h(omega*x)
        x1, x2 = x
        m1 = GaussianMoments.First.value
        m2 = GaussianMoments.Second.value
        hs_value = 10 * jnp.array([
            m1 * x2 - m2 * x1**2])
        return hs_value


class Program27(Program):
    def __init__(self):
        super().__init__(
            name="program_27",
            num_variables=3,
            num_equality_constraints=1,
            num_inequality_constraints=3)

    def initial_guess(self) -> np.array:
        x0 = 2 * np.ones(3)
        return x0

    def objective(self, x: jnp.array) -> float:
        f_value = 0.01*(x[0] - 1)**2
        f_value += (x[1] - x[0]**2)**2
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3 = x
        hs_value = jnp.array([x1 + x3**2 + 1])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3 = x
        m1 = GaussianMoments.First.value
        m2 = GaussianMoments.Second.value
        hs_value = jnp.array([
            m1 * x1 + m2 * x3**2 + 1])
        return hs_value


class Program28(Program):
    def __init__(self):
        super().__init__(
            name="program_28",
            num_variables=3,
            num_equality_constraints=1,
            num_inequality_constraints=3)

    def initial_guess(self) -> np.array:
        x0 = np.array([-4., 1., 1.])
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3 = x
        f_value = (x1 + x2)**2 + (x2 + x3)**2
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3 = x
        hs_value = jnp.array([
            x1 + 2*x2 + 3*x3 - 1])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3 = x
        m1 = GaussianMoments.First.value
        hs_value = jnp.array([
            m1*x1 + 2*m1*x2 + 3*m1*x3 - 1])
        return hs_value


class Program42(Program):
    def __init__(self):
        super().__init__(
            name="program_42",
            num_variables=4,
            num_equality_constraints=2,
            num_inequality_constraints=4)

    def initial_guess(self) -> np.array:
        x0 = np.ones(4)
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4 = x
        f_value = ((x1 - 1)**2 
                   + (x2 - 2)**2
                   + (x3 - 3)**2
                   + (x4 - 4)**2)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4 = x
        hs_value = jnp.array([
            x1 - 2,
            x3**2 + x4**2 - 2])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 3 * jnp.ones_like(x)
        gu = 3 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4 = x
        m1 = GaussianMoments.First.value
        m2 = GaussianMoments.Second.value
        hs_value = jnp.array([
            m1*x1 - 2,
            m2*x3**2 + m2*x4**2 - 2])
        return hs_value


class Program47(Program):
    def __init__(self):
        super().__init__(
            name="program_47",
            num_variables=5,
            num_equality_constraints=3,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = np.array([
            2, np.sqrt(2), -1, 2-np.sqrt(2), 0.5])
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((x1 - x2)**2 
                   + (x2 - x3)**3
                   + (x3 - x4)**4
                   + (x4 - x5)**4)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            x1 + x2**2 + x3**3 - 3,
            x2 - x3**2 + x4 - 1,
            x1*x5 - 1])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        m2 = GaussianMoments.Second.value
        m3 = GaussianMoments.Third.value
        hs_value = jnp.array([
            m1*x1 + m2*x2**2 + m3*x3**3 - 3,
            m1*x2 - m2*x3**2 + m1*x4 - 1,
            m1**2*x1*x5 - 1])
        return hs_value


class Program48(Program):
    def __init__(self):
        super().__init__(
            name="program_48",
            num_variables=5,
            num_equality_constraints=2,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = np.array([
            3., 5., -3., 2., -2.])
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((x1 - 1)**2 
                   + (x2 - x3)**2
                   + (x4 - x5)**2)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            jnp.sum(x) - 5,
            x3 - 2*(x4 + x5) + 3])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        hs_value = jnp.array([
            m1*jnp.sum(x) - 5,
            m1*x3 - 2*(m1*x4 + m1*x5) + 3])
        return hs_value


class Program49(Program):
    def __init__(self):
        super().__init__(
            name="program_49",
            num_variables=5,
            num_equality_constraints=2,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = np.array([
            10, 7, 2, -3, 0.8])
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((x1 - x2)**2 
                   + (x3 - 1)**2
                   + (x4 - 1)**4
                   + (x5 - 1)**6)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            x1 + x2 + x3 + 4*x4 - 7,
            x3 + 5*x5 - 6])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        hs_value = jnp.array([
            m1*x1 + m1*x2 + m1*x3 + 4*m1*x4 - 7,
            m1*x3 + 5*m1*x5 - 6])
        return hs_value


class Program50(Program):
    def __init__(self):
        super().__init__(
            name="program_50",
            num_variables=5,
            num_equality_constraints=3,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = np.array([
            35., -31., 11., 5., -5.])
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((x1 - x2)**2 
                   + (x2 - x3)**2
                   + (x3 - x4)**4
                   + (x4 - x5)**2)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            x1 + 2*x2 + 3*x3 - 6,
            x2 + 2*x3 + 3*x4 - 6,
            x3 + 2*x4 + 3*x5 - 6])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        hs_value = jnp.array([
            m1*x1 + 2*m1*x2 + 3*m1*x3 - 6,
            m1*x2 + 2*m1*x3 + 3*m1*x4 - 6,
            m1*x3 + 2*m1*x4 + 3*m1*x5 - 6])
        return hs_value


class Program51(Program):
    def __init__(self):
        super().__init__(
            name="program_51",
            num_variables=5,
            num_equality_constraints=3,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = np.array([
            2.5, 0.5, 2.0, -1.0, 0.5])
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((x1 - x2)**2 
                   + (x2 + x3 - 2)**2
                   + (x4 - 1)**2
                   + (x5 - 1)**2)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            x1 + 3*x2 - 4,
            x3 + x4 - 2*x5,
            x2 - x5])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        hs_value = jnp.array([
            m1*x1 + 3*m1*x2 - 4,
            m1*x3 + m1*x4 - 2*m1*x5,
            m1*x2 - m1*x5])
        return hs_value


class Program52(Program):
    def __init__(self):
        super().__init__(
            name="program_52",
            num_variables=5,
            num_equality_constraints=3,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = 2 * np.ones(5)
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((4*x1 - x2)**2 
                   + (x2 + x3 - 2)**2
                   + (x4 - 1)**2
                   + (x5 - 1)**2)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            x1 + 3*x2,
            x3 + x4 - 2*x5,
            x2 - x5])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        hs_value = jnp.array([
            m1*x1 + 3*m1*x2,
            m1*x3 + m1*x4 - 2*m1*x5,
            m1*x2 - m1*x5])
        return hs_value


class Program61(Program):
    def __init__(self):
        super().__init__(
            name="program_61",
            num_variables=3,
            num_equality_constraints=2,
            num_inequality_constraints=3)

    def initial_guess(self) -> np.array:
        x0 = np.zeros(3)
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3 = x
        f_value = (4*x1**2 
                   + 2*x2**2
                   + 2*x3**2
                   - 33*x1
                   + 16*x2
                   - 24*x3)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3 = x
        hs_value = jnp.array([
            3*x1 - 2*x2**2 - 7,
            4*x1 - x3**2 - 11])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 10 * jnp.ones_like(x)
        gu = 10 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3 = x
        m1 = GaussianMoments.First.value
        m2 = GaussianMoments.Second.value
        hs_value = jnp.array([
            3*m1*x1 - 2*m2*x2**2 - 7,
            4*m1*x1 - m2*x3**2 - 11])
        return hs_value


class Program79(Program):
    def __init__(self):
        super().__init__(
            name="program_79",
            num_variables=5,
            num_equality_constraints=3,
            num_inequality_constraints=5)

    def initial_guess(self) -> np.array:
        x0 = 2 * np.ones(5)
        return x0

    def objective(self, x: jnp.array) -> float:
        x1, x2, x3, x4, x5 = x
        f_value = ((x1 - 1)**2 
                   + (x1 - x2)**2
                   + (x2 - x3)**2
                   + (x3 - x4)**4
                   + (x4 - x5)**4)
        return f_value

    def equality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        hs_value = jnp.array([
            x1 + x2**2 + x3**2 - 2 - 3*jnp.sqrt(2),
            x2 - x3**2 + x4 + 2 - 2*jnp.sqrt(2),
            x1*x5 - 2])
        return hs_value

    def inequality_constraints(
            self, 
            x: jnp.array) -> jnp.array:
        gx = x
        gl = - 2 * jnp.ones_like(x)
        gu = 2 * jnp.ones_like(x)
        return gx, gl, gu

    def equality_constraints_average_value(
            self, 
            x: jnp.array) -> jnp.array:
        x1, x2, x3, x4, x5 = x
        m1 = GaussianMoments.First.value
        m2 = GaussianMoments.Second.value
        hs_value = jnp.array([
            m1*x1 + m2*x2**2 + m2*x3**2 - 2 - 3*jnp.sqrt(2),
            m1*x2 - m2*x3**2 + m1*x4 + 2 - 2*jnp.sqrt(2),
            m1**2*x1*x5 - 2])
        return hs_value


if __name__=="__main__":
    print("[benchmark_optimization_problems.py]")
    Program(
        name="Test",
        num_variables=1,
        num_equality_constraints=1,
        num_inequality_constraints=1)
    Program6().test()
    SampledProgram(
        Program6(),
        sample_size=100).test()
    Program6().test()
    Program27().test()
    Program28().test()
    Program42().test()
    Program47().test()
    Program48().test()
    Program49().test()
    Program50().test()
    Program51().test()
    Program52().test()
    Program61().test()
    Program79().test()
    print("Test successful.")
