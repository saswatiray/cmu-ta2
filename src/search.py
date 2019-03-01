
import os, sys, logging
import util
import solution_templates
import problem_pb2
import primitive_lib
from multiprocessing import Pool, cpu_count

def search_phase():
    """
    TA2 running in stand-alone search phase
    """
    inputDir = os.environ['D3MINPUTDIR']
    outputDir = os.environ['D3MOUTPUTDIR']
    timeout_env = os.environ['D3MTIMEOUT']
    num_cpus = 8 #os.environ['D3MCPU']
    problemPath = os.environ['D3MPROBLEMPATH']

    logging.info("D3MINPUTDIR = %s", inputDir)
    logging.info("D3MOUTPUTDIR = %s", outputDir)
    logging.info("timeout = %s", timeout_env)
    logging.info("cpus = %s", num_cpus)
    (dataset, task_name, problem_desc) = util.load_data_problem(inputDir, problemPath)

    timeout_in_min = (int)(timeout_env)
    primitives = primitive_lib.load_primitives()
    task_name = task_name.upper()
    logging.info(task_name)

    solutions = solution_templates.get_solutions(task_name, dataset, primitives, None)

    async_message_thread = Pool((int)(num_cpus))
    valid_solutions = {}
    valid_solution_scores = {}

    inputs = []
    inputs.append(dataset)
    if task_name == 'REGRESSION':
        metric= problem_pb2.MEAN_SQUARED_ERROR
    else:
        metric= problem_pb2.F1_MACRO

    # Score potential solutions
    results = [async_message_thread.apply_async(evaluate_solution_score, (inputs, sol, primitives, metric,)) for sol in solutions]
    timeout = timeout_in_min * 60
    halftimeout = None
    if timeout <= 0:
        timeout = None
    elif timeout > 60:
        timeout = timeout - 60

    if timeout is not None:
        halftimeout = timeout/2

    index = 0
    for r in results:
        try:
            (score, optimal_params) = r.get(timeout=halftimeout)
            id = solutions[index].id
            valid_solutions[id] = solutions[index]
            valid_solution_scores[id] = score
            if optimal_params is not None and len(optimal_params) > 0:
                valid_solutions[id].set_hyperparams(optimal_params)
        except:
            logging.info(solutions[index].primitives)
            logging.info(sys.exc_info()[0])
            logging.info("Solution terminated: %s", solutions[index].id)
        index = index + 1

    # Sort solutions by their scores and rank them
    import operator
    sorted_x = sorted(valid_solution_scores.items(), key=operator.itemgetter(1))
    if util.invert_metric(metric) is False:
        sorted_x.reverse()

    index = 1
    for (sol, score) in sorted_x:
        valid_solutions[sol].rank = index
        print("Rank ", index)
        print("Score ", score)
        print(valid_solutions[sol].primitives)
        index = index + 1

    num = 20
    if len(sorted_x) < 20:
        num = len(sorted_x)

    util.initialize_for_search(outputDir)

    # Fit solutions and dump out files
    sorted_x = sorted_x[:num]
    results = [async_message_thread.apply_async(fit_solution, (inputs, valid_solutions[sol], primitives, outputDir, problem_desc))
     for (sol,score) in sorted_x]

    index = 0
    for r in results:
        try:
            valid=r.get(timeout=halftimeout)
        except:
            logging.info(valid_solutions[sorted_x[index][0]].primitives)
            logging.info(sys.exc_info()[0])
            logging.info("Solution terminated: %s", valid_solutions[sorted_x[index][0]].id)
        index = index + 1

def evaluate_solution_score(inputs, solution, primitives, metric):
    """
    Scores each potential solution
    Runs in a separate process
    """
    logging.info("Evaluating %s", solution.id)

    (score, optimal_params) = solution.score_solution(inputs=inputs, metric=metric,
                                primitive_dict=primitives, solution_dict=None)

    return (score, optimal_params)

def fit_solution(inputs, solution, primitives, outputDir, problem_desc):
    """
    Fits each potential solution
    Runs in a separate process
    """
    logging.info("Fitting %s", solution.id)
    solution.fit(inputs=inputs, solution_dict=None)

    util.write_pipeline_json(solution, primitives, outputDir + "/pipelines_ranked", rank=solution.rank)
    #util.write_pipeline_yaml(solution, outputDir + "/pipeline_runs", inputs, problem_desc)
    return True
