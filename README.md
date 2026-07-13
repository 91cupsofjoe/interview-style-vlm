# General Markdown file items (DELETE THIS AFTER COMPLETION!)

1. Project name
2. Very short (1-2 lines) description
3. Features (list, what user can accomplish -- NOT listing classes, modules, etc.)
4. Installation guide
5. Example user implementation
6. Project structure
7. Licenses

## Implementation

# model.model.py:
    - load(): Loads model configurations and data from an input file.
    - save(): Loads model configurations and data to an input file.

    - set_dataset(): Sets the DataSet object as the model dataset.
    [* NOTE *] DataSet is a subclass of Dataset (from torch.utils.data)

    - train(): Train the model and return a boolean indicating training
        success. Optional keyword arguments include:
        num_epochs, batch_size, eps (loss threshold), patience (threshold for
        # of times eps is reached), and more.
        [* NOTE *] Can provide input files, sample data lists, or training/test
        tensors. If input files or sample data is provided, they are set in the
        model before use.

    - predict(): Returns the model prediction for the query example.

    [** NOTE **] model.py contains wrapper functions for its dataset methods
    [see below].

# data.dataset.py:
    - get_training_data(): Returns training examples and training labels.
    - get_test_data(): Returns test examples and test labels.
        [* NOTE *] Both these methods return lists unless use_tensors is set to
        true.

    - load_training_data(): Loads input files or lists into the dataset.
    - load_test_data(): Loads input files or lists into the dataset.
        [* NOTE *] Both these methods require a tensor function pointer or name
        for load_data() [see below].

    - load_data(): Loads input files or lists into the dataset.
        [* NOTE *] This method requires a tensor function pointer or name used
        to convert sample data lists into tensors.
        [** NOTE **] Use training_test_split to separate training and test
        data. Floats less than 0 or greater than 1 will be treated as integer
        indices (standard array slicing), while floats greater than 0 but less
        than 1 will be treated as fractional indices (used to determine array
        slicing).