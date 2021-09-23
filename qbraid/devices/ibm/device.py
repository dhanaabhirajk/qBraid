"""QiskitBackendWrapper Class"""

from qiskit import Aer, BasicAer, IBMQ, execute
from qiskit.providers.ibmq import least_busy
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.providers.backend import Backend as QiskitBackend

from qbraid.devices._utils import get_config
from qbraid.devices.device import DeviceLikeWrapper
from qbraid.devices.ibm.job import QiskitJobWrapper
from qbraid.devices.ibm.result import QiskitResultWrapper
from qbraid.devices.exceptions import DeviceError


class QiskitBackendWrapper(DeviceLikeWrapper):
    """Wrapper class for IBM Qiskit ``Backend`` objects."""

    def __init__(self, device_info, **fields):
        """Create a QiskitBackendWrapper

        Args:
            name (str): a Qiskit supported device
            provider (str): the provider that this device comes from
            fields: kwargs for the values to use to override the default options.

        Raises:
            DeviceError: if input field not a valid options

        """
        super().__init__(device_info, **fields)

    def _get_device(self, obj_ref, obj_arg) -> QiskitBackend:
        """Initialize an IBM device."""
        if obj_ref == "IBMQ":
            if IBMQ.active_account() is None:
                IBMQ.load_account()
            group = get_config("group", "IBM")
            project = get_config("project", "IBM")
            provider = IBMQ.get_provider(hub="ibm-q", group=group, project=project)
            if obj_arg == "least_busy":
                backends = provider.backends(filters=lambda x: not x.configuration().simulator)
                return least_busy(backends)
            else:
                return provider.get_backend(obj_arg)
        elif obj_ref == "Aer":
            return Aer.get_backend(obj_arg)
        elif obj_ref == "BasicAer":
            return BasicAer.get_backend(obj_arg)
        else:
            raise DeviceError(f"obj_ref {obj_ref} not found.")

    def execute(self, run_input, *args, **kwargs):
        """Runs circuit(s) on qiskit backend via :meth:`~qiskit.utils.QuantumInstance.execute`.

        Creates a :class:`~qiskit.utils.QuantumInstance`, invokes its ``execute`` method,
        applies a QiskitResultWrapper, and returns the result. Internally, the quantum instance
        execute method used :meth:`~qiskit.utils.QuantumInstance.transpile` to ensure that qiskit
        algorithms can access the given circuit(s).

        Args:
            run_input: An individual or a list of circuit objects to run on the wrapped device.
            kwargs: Any kwarg options to pass to the device for the run. If a key is also present
                in the options attribute/object then the expectation is that the value specified
                will be used instead of what's set in the options object.

        Returns:
            QiskitResultWrapper: The result like object for the run.

        """
        run_input = self._compat_run_input(run_input)
        quantum_instance = QuantumInstance(self.vendor_dlo, *args, **kwargs)
        qiskit_result = quantum_instance.execute(run_input)
        qbraid_result = QiskitResultWrapper(qiskit_result)
        return qbraid_result

    def run(self, run_input, *args, **kwargs) -> QiskitJobWrapper:
        """Runs circuit(s) on qiskit backend via :meth:`~qiskit.execute`

        Uses the :meth:`~qiskit.execute` method to create a :class:`~qiskit.providers.Job` object,
        applies a :class:`~qbraid.devices.ibm.QiskitJobWrapper`, and return the result. Depending
        on the backend this may be either an async or sync call. It is the discretion of the
        provider to decide whether running should block until the execution is finished or not.
        The qiskit Job class can handle either situation.

        Args:
            run_input: An individual or a list of circuit objects to run on the wrapped device.
            kwargs: Any kwarg options to pass to the backend for running the config. If a key is
                also present in the options attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options object.
        Returns:
            QiskitJobWrapper: The job like object for the run.

        """
        run_input = self._compat_run_input(run_input)
        qiskit_device = self.vendor_dlo
        qiskit_job = execute(run_input, qiskit_device, *args, **kwargs)
        qbraid_job = QiskitJobWrapper(self, qiskit_job)
        return qbraid_job
