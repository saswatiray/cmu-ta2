import unittest
from api_v1 import core, data_ext, dataflow_ext


import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc

from concurrent import futures
import grpc

import util
import bo
import docker


class TestCore(unittest.TestCase):
    def setUp(self):
        # self.client = docker.from_env()
        # self.client.containers.run("8546fc3ed6c5")
        return
        threadpool = futures.ThreadPoolExecutor(max_workers=4)
        self.__server__ = grpc.server(threadpool)
        core.add_to_server(self.__server__)
        data_ext.add_to_server(self.__server__)
        dataflow_ext.add_to_server(self.__server__)
        self.__server__.add_insecure_port('localhost:45042')
        self.__server__.start()

    def tearDown(self):
        # c = self.client.containers.get('8546fc3ed6c5')
        # c.stop()
        return
        self.__server__.stop(0)

    def test_session(self):
        channel = grpc.insecure_channel('localhost:45042')
        stub = core_pb2_grpc.CoreStub(channel)
        msg = core_pb2.SessionRequest(user_agent="unittest", version="Foo")
        session = stub.StartSession(msg)
        self.assertTrue(session.response_info.status.code == core_pb2.OK)

        session_end_response = stub.EndSession(session.context)
        self.assertTrue(session_end_response.status.code == core_pb2.OK)

        # Try to end a session that does not exist
        fake_context = core_pb2.SessionContext(session_id="fake context")
        session_end_response = stub.EndSession(fake_context)
        self.assertTrue(session_end_response.status.code == core_pb2.SESSION_UNKNOWN)


    def test_pipeline(self):
        "Tries setting up a new pipeline"
        channel = grpc.insecure_channel('localhost:45042')
        stub = core_pb2_grpc.CoreStub(channel)
        msg = core_pb2.SessionRequest(user_agent="unittest", version="Foo")
        session = stub.StartSession(msg)
        self.assertTrue(session.response_info.status.code == core_pb2.OK)

        pipeline_request = core_pb2.PipelineCreateRequest(
            context=session.context,
            dataset_uri="file:///home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json",
            task=core_pb2.TASK_TYPE_UNDEFINED,
            task_subtype=core_pb2.TASK_SUBTYPE_UNDEFINED,
            task_description="",
            output=core_pb2.OUTPUT_TYPE_UNDEFINED,
            metrics=[],
            target_features=[],
            predict_features=[],
            max_pipelines=10
        )
        p = stub.CreatePipelines(pipeline_request)
        for response in p:
            self.assertTrue(response.response_info.status.code == core_pb2.OK)



import problem
import core_pb2
class TestProblemSolver(unittest.TestCase):
    def test_solution_finding(self):
        # tasktype = core_pb2.TaskType
        # print(tasktype.Value('CLASSIFICATION'))
        # print(type(tasktype.Name(1)))
        p = problem.ProblemDescription("test", "file:///home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json", "/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/output", core_pb2.CLASSIFICATION, [], [], [])
        for pipeline in p.find_solutions():
            pipeline.train("file:///home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json")
            pipeline.evaluate("file:///home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TEST/dataset_TEST/datasetDoc.json")

            # import numpy as np
            # import bo
            # from bo.utils.function_caller import get_function_caller_from_function

            # domain_bounds = np.array([])
            # caller = get_function_caller_from_function(pipeline.evaluate, domain_bounds, False)
            # need domain bounds, hyperparameter bounds


import numpy as np
from bo.gp import gp_core, gp_instances, kernel
class TestBO(unittest.TestCase):
    def test_bayesian_opt(self):
        """
        Simple tests to ensure that the library loads and runs correctly.
        """

        def get_data():
            """ Generates data. """
            func = lambda t: (-70 * (t-0) * (t-0.35) * (t+0.55) * (t-0.65) * (t-0.97)).sum(axis=1)
            N = 5
            X_tr = np.array(range(N)).astype(float).reshape((N, 1))/N + 1/(float(2*N))
            Y_tr = func(X_tr)
            # kern = kernel.SEKernel(1, 1, 0.5)
            kern = kernel.PolyKernel(2, 2, 2, 12)
            data = {"func":func, "X_tr":X_tr, "Y_tr":Y_tr, "kern":kern}
            return data


        def _demo_common(gp, data, desc):
            """ Common processes for the demo. """
            lml = gp.compute_log_marginal_likelihood()
            print(desc + ': Log-Marg-Like: ' + str(lml) + ', kernel: ' + str(gp.kernel.hyperparams))
            gp.visualise(true_func=data['func'], boundary=[0, 1])

        def demo_gp_given_hps(data, kern, desc):
            """ A demo given the kernel hyper-parameters. """
            mean_func = lambda x: np.array([data['Y_tr'].mean()] * len(x))
            noise_var = data['Y_tr'].std()/10
            est_gp = gp_core.GP(data['X_tr'], data['Y_tr'], kern, mean_func, noise_var)
            _demo_common(est_gp, data, desc)

        def demo_gp_fit_hps(data, desc):
            """ A demo where the kernel hyper-parameters are fitted. """
            fitted_gp, _ = gp_instances.SimpleGPFitter(data['X_tr'], data['Y_tr']).fit_gp()
            _demo_common(fitted_gp, data, desc)

        # The SimpleGPFitter reads a lot of options from the command line
        # and we don't want to try to re-write big piles of it
        # so we fudge it so it doesn't get confused by unit tests
        import sys
        sys.argv = ['python']

        data = get_data()
        del data['func']
        print(data)
        print('First fitting a GP with the given kernel. Close window to continue.')
        demo_gp_given_hps(data, data['kern'], 'Given Kernel')
        print('\nNow estimating kernel via marginal likelihood. Close window to continue.')
        demo_gp_fit_hps(data, 'Fitted Kernel')

if __name__ == '__main__':
    unittest.main()
