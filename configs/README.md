# SRZoo Model Config Files

This folder contains the model config files (in JSON format) for pre-trained models provided by SRZoo.
If you converted your own super-resolution model, you may also need to make a config file for the converted model.


## Parameters

- ```channel_first```: (Default: false) Specify this when your model deals with the channels in the first dimension.
- ```input_name```: (Default: "sr_input") Name of the input node (low-resolution image).
- ```input_scale_name```: (Default: "sr_input_scale") Name of the input scale node. Only used if ```use_scale_placeholder``` is set to true.
- ```output_name```: (Default: "sr_output") Name of the output node (super-resolved image).
- ```pixel_range```: (Default: [0.0, 255.0]) Range of the pixel values that the model uses.
- ```use_scale_placeholder```: (Default: false) Specify this when your model requires inputting the current upscaling factor.

