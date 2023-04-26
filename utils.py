import re
def parse_action_line(input_string):
    action_regex = r"Observation:.+Thought:.+Action: (.+)"
    match = re.search(action_regex, input_string, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return "Action tag not found in the input string."
