# Utils package for functional modeling

# Import functions from the parent utils.py file
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Import from the standalone utils.py file
import importlib.util
utils_path = os.path.join(parent_dir, 'utils.py')
spec = importlib.util.spec_from_file_location("utils_module", utils_path)
utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_module)

# Export the function
complex_format_fn = utils_module.complex_format_fn

__all__ = ['complex_format_fn']
