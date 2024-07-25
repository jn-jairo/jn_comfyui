from ..utils import CATEGORY_WORKFLOW

class JN_Condition:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("*",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "condition": ("BOOLEAN", {"default": True}),
                "if_true": ("*",),
                "if_false": ("*",),
            },
        }

    def run(self, condition=True, if_true=True, if_false=False):
        value = if_true if condition else if_false
        return (value,)

class JN_StopIf:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "condition": ("BOOLEAN", {"default": False}),
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "flow": ("*",),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
            },
        }

    def run(self, condition=False, title="", flow=None, dependency=None):
        if condition:
            raise Exception(title)

        return (flow,)

class JN_StopIfOutput:
    CATEGORY = CATEGORY_WORKFLOW
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "condition": ("BOOLEAN", {"default": False}),
                "title": ("STRING", {"default": "", "dynamicPrompts": False}),
                "dependency": ("*", {"multiple": True}),
            },
            "required": {
            },
        }

    def run(self, condition=False, title="", dependency=None):
        if condition:
            raise Exception(title)

        return {"ui": {}}

NODE_CLASS_MAPPINGS = {
    "JN_Condition": JN_Condition,
    "JN_StopIf": JN_StopIf,
    "JN_StopIfOutput": JN_StopIfOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JN_Condition": "Condition",
    "JN_StopIf": "Stop If",
    "JN_StopIfOutput": "Stop If Output",
}
