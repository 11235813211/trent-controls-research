import numpy as np
from typing import Callable, Any, Tuple, List

class ClosedStateSpaceModel():
    def __init__(self, 
            plant_model: Callable[[np.ndarray, np.ndarray], np.ndarray], 
            controller: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],Tuple[np.ndarray, Any]],
            output: Callable[[np.ndarray, np.ndarray], np.ndarray], 
            plant_state: List = None,
            control_state_0: np.ndarray = None, 
            gains: np.ndarray = None):
        """
        Docstring for __init__
        
        :param plant_model: takes state, input and returns derivative of state
        :param controller: takes state, state_ref, control_state, gains and returns input, control_state_new
        :param output: takes state, input and returns output_state
        :param control_state_0: initial control state
        :param gains: gains for the controller
        """
        
        self._state_derivative: Callable[[np.ndarray, np.ndarray, List], np.ndarray] = plant_model
        self._controller_func: Callable[[np.ndarray, np.ndarray, List, List],Tuple[np.ndarray, List]] = controller
        self.output: Callable[[np.ndarray, np.ndarray], np.ndarray] = output

        self._plant_state: List = plant_state

        self._control_state: List = control_state_0
        self._control_state_0: List = control_state_0

        self._gains: np.ndarray = gains
    
    def controller(self, state_: np.ndarray, state_ref: np.ndarray) -> np.ndarray:
        input_, control_state_ = self._controller_func(state_, state_ref, self._control_state, self._gains)
        
        self._control_state = control_state_
        return input_
    
    def state_derivative(self, state_: np.ndarray, input_: np.ndarray) -> np.ndarray:
        return self._state_derivative(state_, input_, self._plant_state)
    
    def initialize(self, gains = None):
        self._control_state = self._control_state_0.copy()
        if gains != None:
            self._gains = gains

