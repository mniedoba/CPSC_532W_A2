# Standard imports
import torch
from enum import Enum
from collections import defaultdict

# Project imports
from primitives import primitives

class EvaluationScheme(Enum):

    DEFAULT = 0
    IS = 1
    MH = 2
    GIBBS = 3

class ExpressionType(Enum):

    CONSTANT = 0
    VARIABLE = 1
    LET_BLOCK = 2
    IF_BLOCK = 3
    LOCAL_FNC = 4
    PRIMITIVE_FNC = 5
    SAMPLE = 6
    OBSERVE = 7

class Expression:

    @classmethod
    def get_type(cls, json):
        if json == []:
            raise ValueError("Empty json has not type.")
        elif not isinstance(json, list):
            # Zero length json is either a constant or a variable name.
            val = json
            # Assume string constants will be surrounded by "" or ''.
            if isinstance(val, str) and len(val) and val[0] != "\"" and val[0] != "\'":
                return ExpressionType.VARIABLE
            else:
                return ExpressionType.CONSTANT
        elif json[0] == 'let':
            # Let blocks start with keyword let.
            return ExpressionType.LET_BLOCK
        elif json[0] == 'if':
            # If block starts with keyword if.
            return ExpressionType.IF_BLOCK
        elif json[0] == 'sample' or json[0] == 'sample*':
            # Sample expression starts with keyword sample.
            return ExpressionType.SAMPLE
        elif json[0] == 'observe' or json[0] == 'observe*':
            # Observe expression starts with keyword observe.
            return ExpressionType.OBSERVE
        elif json[0] in primitives:
            # Last two cases are function calls. If it's in the list of primitive functions, its primitive.
            return ExpressionType.PRIMITIVE_FNC
        else:
            # Otherwise its a defined procedure.
            return ExpressionType.LOCAL_FNC

    def __init__(self, ast_json):
        self.json = ast_json
        self.type = Expression.get_type(self.json)

        # For non-unary expressions, drop the keywords and treat the remaining json elements as expressions.
        if self.type != ExpressionType.CONSTANT and self.type != ExpressionType.VARIABLE:
            self.sub_expressions = [Expression(sym) for sym in self.json[1:]]
        else:
            self.sub_expressions = []

    def __getitem__(self, idx):
        # Helper to index into the sub expressions.
        return self.sub_expressions[idx]

class Procedure:

    def __init__(self, json):
        if ['['] in json:
            raise Exception

        # Procedure consists of a list of variable names to bind to and a list of expressions.
        self.variable_names = json[2]
        self.expression = Expression(json[-1])

class AbstractSyntaxTree:
    def __init__(self, ast_json):
        self.ast_json = ast_json

        # Global level function
        self.procedures = {}

        # Tree of expressions.
        self.expressions = []
        # Iterate over top components.
        for elem in ast_json:
            if elem[0] == 'defn':
                self.procedures[elem[1]] = Procedure(elem)
            else:
                self.expressions.append(Expression(elem))


def eval_expression(expression, sigma, local_env, procedures, eval_scheme=EvaluationScheme.DEFAULT):
    """Helper fucnction which evaluates expression, using a local environment and a sigma."""
    expr_type = expression.type

    match expr_type:
        case ExpressionType.CONSTANT:
            # Constants are just converted to float tensors.
            return torch.tensor(expression.json, dtype=torch.float), sigma
        case ExpressionType.VARIABLE:
            # Grab the variable from the local environment.
            return local_env[expression.json][-1], sigma
        case ExpressionType.LET_BLOCK:
            # Let is a definition expression (in Daphne) along with an expression.
            definition, sub_expr = expression[0], expression[1]
            # Get the variable name and expression for the definition.
            var_name, var_expr = definition.json[0], definition[0]
            # Get the value for the variable and assign to the variable name.
            var_value, new_sigma = eval_expression(var_expr, sigma, local_env, procedures)
            local_env[var_name].append(var_value)
            r_value, r_sigma = eval_expression(sub_expr, new_sigma, local_env, procedures)
            local_env[var_name].pop()
            return r_value, r_sigma
        case ExpressionType.IF_BLOCK:
            # Get the relevant components of the ternary.
            predicate, consequent, antecedent = expression.sub_expressions
            # Evaluate the condition
            predicate_value, new_sigma = eval_expression(predicate, sigma, local_env, procedures)
            # Evaluate the sub expression based on the predicate value.
            return eval_expression(consequent if predicate_value else antecedent, sigma, local_env, procedures)
        case ExpressionType.SAMPLE:
            dist_obj, new_sigma = eval_expression(expression[0], sigma, local_env, procedures)
            return dist_obj.sample(), new_sigma
        case ExpressionType.OBSERVE:
            match eval_scheme:
                case EvaluationScheme.IS:
                    assert len(expression.sub_expressions) == 2
                    e1, e2 = expression.sub_expressions
                    d1, sigma = eval_expression(e1, sigma, local_env, eval_scheme)
                    c2, sigma = eval_expression(e2, sigma, local_env, eval_scheme)
                    log_w = d1.log_prob(c2)
                    sigma['log_w'] += log_w
                    return c2, sigma
                case other:
                    return None, sigma
        case other:
            # Handles function calls.
            values = []
            # Get the values which will be bound to function arguments.
            for sub_expr in expression.sub_expressions:
                value, sigma = eval_expression(sub_expr, sigma, local_env, procedures)
                values.append(value)
            if expr_type == ExpressionType.LOCAL_FNC:
                procedure = procedures[expression.json[0]]
                for variable_name, value in zip(procedure.variable_names, values):
                    local_env[variable_name].append(value)
                r_value, r_sigma = eval_expression(procedure.expression, sigma, local_env, procedures)
                for variable_name in procedure.variable_names:
                    local_env[variable_name].pop()
                return r_value, r_sigma
            else:
                return primitives[expression.json[0]](*values), sigma


def evaluate_program(ast, verbose=False):

    local_env = defaultdict(list)
    sigma = {}
    vals = []
    for expr in ast.expressions:
        val, sigma = eval_expression(expr, sigma, local_env, ast.procedures)
        vals.append(val)
    if len(vals) == 1:
        return vals[0], sigma, local_env
    else:
        return vals, sigma, local_env