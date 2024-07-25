# JNComfy

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension with patches and nodes.

## Patches

The patches are applied automatically, some features require configuration,
copy the [jncomfy.yaml.example](jncomfy.yaml.example) to `ComfyUI/jncomfy.yaml`
and uncomment the configuration for the feature you need.

### Preview device

Allows to change the device used for TAESD to render the preview.

The default behaviour is to use the same device used to render the image,
so the preview consumes VRAM that could be used to render the image.

This feature is useful if you don't have enough VRAM to render the preview
and image on the same device.

```yaml
preview_device: cpu
```

### Extension device

Allows to change the device used by custom nodes.

Custom nodes use the `comfy.model_management.get_torch_device()` to get the device
they should use, it is the same device used to render the image,
but some custom nodes performe actions that don't require the same device,
so with this feature you can set another device based on which code is asking for the device.

It matches the package/class that calls the function `comfy.model_management.get_torch_device()` on the custom node.

Example, if it's called at foo.bar.MyCustomNode, any of these will work:

```yaml
extension_device:
  foo: cpu
  foo.bar: cpu
  foo.bar.MyCustomNode: cpu
```

The first part is always the package/repository name, like in this real example:

```yaml
extension_device:
  comfyui_controlnet_aux: cpu
  jn_comfyui.nodes.facerestore: cpu
```

It is easy to change the device for all custom nodes from the same repository,
just use the directory name inside the `custom_nodes` directory.

If the custom nodes are inside `custom_nodes/some_custom_nodes_package` you can set:

```yaml
extension_device:
  some_custom_nodes_package: cpu
```

But to specify specific nodes you need to know how the code of the custom node works
and where it calls the `comfy.model_management.get_torch_device()`.

Depending of how the custom node works it may not be possible to specify just a specific node.

### Temperature

If your device don't have a good cooling system it can overheat after too many consecutive generations.

With this feature you can configure temperature limits to pause the generation and wait the device cool down.

You can set a limit to pause on the `execution` process between the nodes
and another limit for the `progress` process between the steps of a node, for the nodes the show the progress bar.

Each limit has a `safe` and `max` temperature, once the temperature exceeds the `max` temperature
it pauses the generation and waits for it to cool down to the `safe` temperature.

You can set how many seconds it waits before checking the temperature again, the seconds to wait is shown in the progress bar on the node that is waiting.

You can also set for how long it can wait, if you set it to zero it will wait for how long it needs to reach the safe temperature.

```yaml
temperature:
  limit:
    execution: # Don't execute a node if the temperature is above the max, but wait cool down to the safe temperature.
      safe: 90
      max: 95
    progress: # Don't execute the next step of a node if the temperature is above the max, but wait cool down to the safe temperature.
      safe: 90
      max: 95
  cool_down:
    each: 5 # Seconds to wait the temperature cool down before checking the temperature again.
    max: 0 # Max seconds to wait the temperature cool down.
```

### Memory estimation

The split attention, as the name suggests, splits the data used by the attention process in chunks
and processes each chunk one after another.

The amount of chunks depends of how much VRAM is available,
but the exact amount of memory required for the attention process depends of so many things that we can only estimate that value.

The size of the tensor is used as the base to calculate the memory required and that is multiplied by a value.

You can change that multiplier with the setting:

```yaml
memory_estimation_multiplier: 1
```

### Optimizations

Some features, like split attention and tiled VAE encode/decode, divide the process in steps.

This patch optimizes these features to find the best amount of steps for each process that fits your device
and caches that value so the next generations will run faster.

It helps if you have low VRAM and enables you to generate bigger images.

If you have a good GPU it changes nothing and it won't slow down your generations.


### Easy generic inputs

Some nodes require inputs of any type, there are some hacks out there to do it, but the LiteGraph,
which is the javascript library used to create the graphs already has the generic type `*`.

This patch just finishes the integration of the generic type that already exists,
so you don't have to do any fancy trick, just use the type `*`.

```python
class PrintValue:
    CATEGORY = "_for_testing"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("*",),
            },
        }

    def run(self, value):
        print(value)
        return {"ui": {}}
```

### Easy multiple inputs

Some nodes require multiple inputs of the same type, a common approach to solve this problem is to add two inputs of the same type,
output the result and use that output as the input of a copy of the same node, thus concatenating the results.
Another common approach is to add an arbitrary number of inputs of the same type, usually 4 or 5, and hope it is enough.

That may do the job but it is not a good solution, a better solution is to add new inputs dynamically when you connect the input.

Some custom nodes already do it, but they do it as a hack on specific nodes and they cannot mix it with static inputs.

This patch allows any node to have multiple inputs of the same type that work alongside regular inputs.

It is easy to turn an input into multiple inputs, just add the `"multiple": True` option and the value will be an array of the type.

```python
class ImageGrid:
    CATEGORY = "_for_testing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"multiple": True}),
            },
        }

    def run(self, images):
        # receive the images as an array
        for image in images:
            # ... rest of the code ...
        return (image_grid,)
```

## License

The MIT License (MIT). Please see [License File](LICENSE.md) for more information.
