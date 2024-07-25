from execution import *
import execution
import comfy.utils
import re

def patchExecution(config):
    def get_input_data(inputs, class_def, unique_id, outputs={}, prompt={}, extra_data={}):
        valid_inputs = class_def.INPUT_TYPES()
        input_data_all = {}
        for x in inputs:
            input_data = inputs[x]
            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                output_index = input_data[1]
                if input_unique_id not in outputs:
                    input_data_all[x] = (None,)
                    continue
                obj = outputs[input_unique_id][output_index]
                input_data_all[x] = obj
            else:
                if ("required" in valid_inputs and x in valid_inputs["required"]) or ("optional" in valid_inputs and x in valid_inputs["optional"]):
                    input_data_all[x] = [input_data]

        if "hidden" in valid_inputs:
            h = valid_inputs["hidden"]
            for x in h:
                if h[x] == "PROMPT":
                    input_data_all[x] = [prompt]
                if h[x] == "EXTRA_PNGINFO":
                    input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
                if h[x] == "UNIQUE_ID":
                    input_data_all[x] = [unique_id]

        input_data_all_remove = []
        input_data_all_add = {}

        for x in input_data_all:
            m = re.search("(.+)\[(\d*)\]$", x)
            if m:
                input_data_all_remove.append(x)
                x_multiple = m.group(1)
                if x_multiple not in input_data_all_add:
                    input_data_all_add[x_multiple] = [[] for v in input_data_all[x]]
                for k, v in enumerate(input_data_all[x]):
                    input_data_all_add[x_multiple][k].append(v)

        for x in input_data_all_remove:
            del input_data_all[x]

        input_data_all.update(input_data_all_add)

        return input_data_all

    def recursive_execute(server, prompt, outputs, current_item, extra_data, executed, prompt_id, outputs_ui, object_storage):
        unique_id = current_item
        inputs = prompt[unique_id]['inputs']
        class_type = prompt[unique_id]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        if unique_id in outputs:
            return (True, None, None)

        for x in inputs:
            input_data = inputs[x]

            if isinstance(input_data, list):
                input_unique_id = input_data[0]
                output_index = input_data[1]
                if input_unique_id not in outputs:
                    result = recursive_execute(server, prompt, outputs, input_unique_id, extra_data, executed, prompt_id, outputs_ui, object_storage)
                    if result[0] is not True:
                        # Another node failed further upstream
                        return result

        input_data_all = None
        try:
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)
            if server.client_id is not None:
                server.last_node_id = unique_id
                server.send_sync("executing", { "node": unique_id, "prompt_id": prompt_id }, server.client_id)

            comfy.utils.wait_cooldown(kind="execution")

            obj = object_storage.get((unique_id, class_type), None)
            if obj is None:
                obj = class_def()
                object_storage[(unique_id, class_type)] = obj

            output_data, output_ui = get_output_data(obj, input_data_all)
            outputs[unique_id] = output_data
            if len(output_ui) > 0:
                outputs_ui[unique_id] = output_ui
                if server.client_id is not None:
                    server.send_sync("executed", { "node": unique_id, "output": output_ui, "prompt_id": prompt_id }, server.client_id)
        except comfy.model_management.InterruptProcessingException as iex:
            logging.info("Processing interrupted")

            # skip formatting inputs/outputs
            error_details = {
                "node_id": unique_id,
            }

            return (False, error_details, iex)
        except Exception as ex:
            typ, _, tb = sys.exc_info()
            exception_type = full_type_name(typ)
            input_data_formatted = {}
            if input_data_all is not None:
                input_data_formatted = {}
                for name, inputs in input_data_all.items():
                    input_data_formatted[name] = [format_value(x) for x in inputs]

            output_data_formatted = {}
            for node_id, node_outputs in outputs.items():
                output_data_formatted[node_id] = [[format_value(x) for x in l] for l in node_outputs]

            logging.error(f"!!! Exception during processing!!! {ex}")
            logging.error(traceback.format_exc())

            error_details = {
                "node_id": unique_id,
                "exception_message": str(ex),
                "exception_type": exception_type,
                "traceback": traceback.format_tb(tb),
                "current_inputs": input_data_formatted,
                "current_outputs": output_data_formatted
            }
            return (False, error_details, ex)

        executed.add(unique_id)

        return (True, None, None)

    def validate_inputs(prompt, item, validated):
        unique_id = item
        if unique_id in validated:
            return validated[unique_id]

        inputs = prompt[unique_id]['inputs']
        class_type = prompt[unique_id]['class_type']
        obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]

        class_inputs = obj_class.INPUT_TYPES()
        required_inputs = class_inputs['required']

        errors = []
        valid = True

        validate_function_inputs = []
        if hasattr(obj_class, "VALIDATE_INPUTS"):
            validate_function_inputs = inspect.getfullargspec(obj_class.VALIDATE_INPUTS).args

        for x in required_inputs:
            missing = True
            vals = []

            for i in inputs:
                if x == i:
                    vals.append(inputs[i])
                    missing = False
                else:
                    m = re.search("(.+)\[(\d*)\]$", i)
                    if m and x == m.group(1) and isinstance(inputs[i], list):
                        vals.append(inputs[i])
                        missing = False

            if missing:
                error = {
                    "type": "required_input_missing",
                    "message": "Required input is missing",
                    "details": f"{x}",
                    "extra_info": {
                        "input_name": x
                    }
                }
                errors.append(error)
                continue

            info = required_inputs[x]
            type_input = info[0]

            for val in vals:
                if isinstance(val, list):
                    if len(val) != 2:
                        error = {
                            "type": "bad_linked_input",
                            "message": "Bad linked input, must be a length-2 list of [node_id, slot_index]",
                            "details": f"{x}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": info,
                                "received_value": val
                            }
                        }
                        errors.append(error)
                        continue

                    o_id = val[0]
                    o_class_type = prompt[o_id]['class_type']
                    r = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES
                    if r[val[1]] != type_input and r[val[1]] != "*" and type_input != "*":
                        received_type = r[val[1]]
                        details = f"{x}, {received_type} != {type_input}"
                        error = {
                            "type": "return_type_mismatch",
                            "message": "Return type mismatch between linked nodes",
                            "details": details,
                            "extra_info": {
                                "input_name": x,
                                "input_config": info,
                                "received_type": received_type,
                                "linked_node": val
                            }
                        }
                        errors.append(error)
                        continue
                    try:
                        r = validate_inputs(prompt, o_id, validated)
                        if r[0] is False:
                            # `r` will be set in `validated[o_id]` already
                            valid = False
                            continue
                    except Exception as ex:
                        typ, _, tb = sys.exc_info()
                        valid = False
                        exception_type = full_type_name(typ)
                        reasons = [{
                            "type": "exception_during_inner_validation",
                            "message": "Exception when validating inner node",
                            "details": str(ex),
                            "extra_info": {
                                "input_name": x,
                                "input_config": info,
                                "exception_message": str(ex),
                                "exception_type": exception_type,
                                "traceback": traceback.format_tb(tb),
                                "linked_node": val
                            }
                        }]
                        validated[o_id] = (False, reasons, o_id)
                        continue
                else:
                    try:
                        if type_input == "INT":
                            val = int(val)
                            inputs[x] = val
                        if type_input == "FLOAT":
                            val = float(val)
                            inputs[x] = val
                        if type_input == "STRING":
                            val = str(val)
                            inputs[x] = val
                    except Exception as ex:
                        error = {
                            "type": "invalid_input_type",
                            "message": f"Failed to convert an input value to a {type_input} value",
                            "details": f"{x}, {val}, {ex}",
                            "extra_info": {
                                "input_name": x,
                                "input_config": info,
                                "received_value": val,
                                "exception_message": str(ex)
                            }
                        }
                        errors.append(error)
                        continue

                    if len(info) > 1:
                        if "min" in info[1] and val < info[1]["min"]:
                            error = {
                                "type": "value_smaller_than_min",
                                "message": "Value {} smaller than min of {}".format(val, info[1]["min"]),
                                "details": f"{x}",
                                "extra_info": {
                                    "input_name": x,
                                    "input_config": info,
                                    "received_value": val,
                                }
                            }
                            errors.append(error)
                            continue
                        if "max" in info[1] and val > info[1]["max"]:
                            error = {
                                "type": "value_bigger_than_max",
                                "message": "Value {} bigger than max of {}".format(val, info[1]["max"]),
                                "details": f"{x}",
                                "extra_info": {
                                    "input_name": x,
                                    "input_config": info,
                                    "received_value": val,
                                }
                            }
                            errors.append(error)
                            continue

                    if x not in validate_function_inputs:
                        if isinstance(type_input, list):
                            if val not in type_input:
                                input_config = info
                                list_info = ""

                                # Don't send back gigantic lists like if they're lots of
                                # scanned model filepaths
                                if len(type_input) > 20:
                                    list_info = f"(list of length {len(type_input)})"
                                    input_config = None
                                else:
                                    list_info = str(type_input)

                                error = {
                                    "type": "value_not_in_list",
                                    "message": "Value not in list",
                                    "details": f"{x}: '{val}' not in {list_info}",
                                    "extra_info": {
                                        "input_name": x,
                                        "input_config": input_config,
                                        "received_value": val,
                                    }
                                }
                                errors.append(error)
                                continue

        if len(validate_function_inputs) > 0:
            input_data_all = get_input_data(inputs, obj_class, unique_id)
            input_filtered = {}
            for x in input_data_all:
                if x in validate_function_inputs:
                    input_filtered[x] = input_data_all[x]

            #ret = obj_class.VALIDATE_INPUTS(**input_filtered)
            ret = map_node_over_list(obj_class, input_filtered, "VALIDATE_INPUTS")
            for x in input_filtered:
                for i, r in enumerate(ret):
                    if r is not True:
                        details = f"{x}"
                        if r is not False:
                            details += f" - {str(r)}"

                        error = {
                            "type": "custom_validation_failed",
                            "message": "Custom validation failed for node",
                            "details": details,
                            "extra_info": {
                                "input_name": x,
                                "input_config": info,
                                "received_value": val,
                            }
                        }
                        errors.append(error)
                        continue

        if len(errors) > 0 or valid is not True:
            ret = (False, errors, unique_id)
        else:
            ret = (True, [], unique_id)

        validated[unique_id] = ret
        return ret

    execution.get_input_data = get_input_data
    execution.recursive_execute = recursive_execute
    execution.validate_inputs = validate_inputs

PATCHES = {
    "70_execution": patchExecution,
}
