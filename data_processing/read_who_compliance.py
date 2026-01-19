import json
def load_who_compliance():
    with open('data/who_compliance_test_set.json', 'r') as f:
        data= json.load(f)
        print
    return data['test_cases']

