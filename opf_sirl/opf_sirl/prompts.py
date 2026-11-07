"""Prompt construction.

The task requirements are taken from the ProOPF evaluation scripts
(ProOPF_benchmark/LLM_test/test_level{1..4}.py, `build_prompt`) so that training and
evaluation ask for exactly the same artefact. The answer format follows SIRL:
reasoning, mathematical model and code in three tagged blocks.
"""
from .data.schema import Sample

SYSTEM_PROMPT = """You are an expert in power system optimization and MATPOWER.
Translate the user's optimal power flow (OPF) request into an executable MATPOWER model.

Answer strictly in the following format:
<think>
Step-by-step analysis: base case, every parameter modification (component, index, column, new value or trend), structural modifications (problem type, extra objective terms, extra constraints) and solver options.
</think>
<model>
The resulting optimization model: decision variables, objective function, constraints, and the modified parameters.
</model>
<matlab>
The complete MATLAB/MATPOWER code, with no placeholders.
</matlab>"""

_SCRIPT_L1 = """Generate a complete, executable MATLAB script that implements the described optimal power flow analysis.

Description:
{nl}

**Requirements:**
1. Generate a complete MATLAB script that can be executed directly
2. Start with `define_constants;` to load MATPOWER constants
3. Load the appropriate case file using `loadcase()`
4. Apply all parameter modifications as described
5. **Configure the solver options as specified. If modifications are needed, use "opf.violation"**
6. Run the optimal power flow using `runopf(mpc, mpopt)`
7. End with `printpf(results);` to display results
8. The code should be ready to run without any modifications"""

_SCRIPT_L3 = """Generate a complete, executable MATLAB script that implements the described optimal power flow analysis with structural modifications.

Description:
{nl}

**Requirements:**
1. Generate a complete MATLAB script that can be executed directly
2. Start with `define_constants;` to load MATPOWER constants
3. Load the appropriate case file using `loadcase()`
4. Apply all parameter modifications as described:
   - Modify bus parameters (PD, QD, VMAX, VMIN, etc.) using proper indexing
   - Modify generator parameters (PMAX, PMIN, QMAX, QMIN, etc.)
   - Modify branch parameters (BR_X, BR_R, RATE_A, etc.)
5. Handle structural modifications:
   - Set problem type: DC OPF or AC OPF using `mpoption('model', 'DC')` or `'AC'`
   - If custom objective functions are needed (e.g., angle difference penalty, voltage deviation penalty, reactive power cost):
     * Use `om = opf_setup(mpc, mpopt)` to set up the optimization model
     * Use `om.add_quad_cost()` to add quadratic cost terms
   - If additional constraints are needed (e.g., reserve requirements, N-1 security):
     * Use `om.add_var()` to add new variables
     * Use `om.add_lin_constraint()` to add linear constraints
   - Solve using `[x, fval, eflag, ~, ~] = om.solve()`
6. Configure solver options as specified:
   - Always set opf.violation if specified in the description
   - Example: `mpopt = mpoption('opf.violation', 1e-8, 'model', 'DC');`
7. Extract and store the final results in a `results` structure:
   - Extract solution variables from x vector using indices
   - Store objective value: `results.f = fval;`
   - Store convergence status: `results.success = (eflag > 0);`
   - Store bus results (VA, VM if AC) and generator results (PG, QG if AC)
8. The code should be ready to run without any modifications
- Do not use placeholders or comments like "% apply modifications here"
- Include all necessary matrix operations for custom costs/constraints"""

_FUNCTION_L24 = """Based on the following scenario description, generate a MATLAB function that runs an optimal power flow analysis for the specified system.

Scenario:
{nl}

Base system: {base_system}
OPF violation tolerance: {opf_violation}
Solver: {solver}

**Function Requirements:**
1. The MATLAB function MUST be named "run_opf"

2. Configure solver options using mpoption(). If modifications are needed, use opf.violation

3. Function parameters must follow these naming conventions:
   - For bus parameters: bus_PARAMETERNAME_BUSID
     Examples: bus_PD_3
   - For generator parameters: gen_PARAMETERNAME_BUSID
     Examples: gen_PMAX_5, gen_QMAX_5, gen_GENCOST_C1_3
   - For branch parameters: branch_PARAMETERNAME_FBUS_TBUS
     Examples: branch_RATE_A_4_7
   Where:
   - PARAMETERNAME is the exact parameter being modified (e.g., PD)
   - BUSID is the bus number
   - FBUS is the from-bus number
   - TBUS is the to-bus number
   List the parameters in the order in which the modifications are mentioned in the scenario.

4. The function should use these parameter names exactly as inputs and apply them to modify the corresponding MATPOWER case structure.

5. **CRITICAL - Parameter Modification Validation:**
   Each parameter modification in the scenario has a specific trend direction:
   - **Increase**: The new value must be GREATER than the original default value
   - **Decrease**: The new value must be LESS than the original default value
   - **Set zero**: The new value must be exactly ZERO
   **YOU MUST add assert statements to validate EVERY parameter modification follows its trend**, e.g.
   `assert(new_param_value < original_value, 'Parameter must decrease according to scenario');`

6. **CRITICAL - Function Syntax:**
   ALL function definitions MUST be properly closed with an 'end' statement.

7. The function must return the MATPOWER `results` structure (with field `f`)."""


def build_user_prompt(sample: Sample) -> str:
    if sample.level == 1:
        return _SCRIPT_L1.format(nl=sample.natural_language)
    if sample.level == 3:
        return _SCRIPT_L3.format(nl=sample.natural_language)
    spec = sample.spec
    solving = spec.get("solving_requirements", {})
    return _FUNCTION_L24.format(
        nl=sample.natural_language,
        base_system=spec.get("base_system", "case14"),
        opf_violation=solving.get("opf_violation", 1e-6),
        solver=solving.get("solver", "MIPS"),
    )


def build_messages(sample: Sample) -> list:
    """Chat messages in the format consumed by verl / TRL / vLLM chat templates."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample)},
    ]
