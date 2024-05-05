Trainer_conditional_DDPM.py :
This file is used for model training. You can set the model's training parameters on line 467. Once executed, the training will begin automatically, and the code will save the weights and generate the learning curve automatically.




sampler_DDPM.py :
This file is the generator of the model(repaint and repaint-mixed), which can use pre-trained weights to generate images based on sketches. The model's generation parameters are set on line 551, and it will start generating directly when run. You can modify the input sketch folder address on line 512, as well as the label required during generation.

On line 412, you can easily switch between the regular "repaint" and "repaint-mixed" algorithms by changing the value of `mixed_value`. When `mixed_value = 0`, it represents the regular repaint algorithm. When `mixed_value` is greater than zero, the mixed module will be applied. It is recommended to set `mixed_value` to 0.6.
