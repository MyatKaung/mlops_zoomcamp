import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Read the notebook
nb = nbformat.read('starter.ipynb', as_version=4)

# Execute the notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '.'}})

# Write the executed notebook
nbformat.write(nb, 'starter_executed.ipynb')

print("Notebook executed successfully!")