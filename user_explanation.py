from copy import deepcopy


class UserExplanation:
    def __init__(self, explaination, ):
        self.lace_explanation = deepcopy(explaination)
        self.instance_explanation = deepcopy(explaination)
        self.id_user_rules = []

    def update_user_rules(self, explaination, user_rule_id):
        self.instance_explanation = deepcopy(explaination)
        self.id_user_rules.append(user_rule_id)
