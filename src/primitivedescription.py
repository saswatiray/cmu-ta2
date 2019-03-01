import pandas as pd
import math
from sklearn import metrics
from sklearn import preprocessing
import bo.gp_call
import problem_pb2
import util

class PrimitiveDescription(object):
    """
    Class representing single primitive.
    Used for optimizing primitive hyper-parameters, doing cross-validation.
    """
    def __init__(self, primitive, primitive_class):
        self.id = primitive_class.id
        self.primitive = primitive
        self.primitive_class = primitive_class

    def score_primitive(self, X, y, metric_type, custom_hyperparams, step_index=0):
        """
        Learns optimal hyperparameters for the primitive
        Returns metric score and optimal parameters.
        """
        python_path = self.primitive.metadata.query()['python_path']

        # Tune hyperparams for classification/regression primitives currently using Bayesian optimization
        if 'hyperparams' in self.primitive.metadata.query()['primitive_code'] and y is not None and 'Find_projections' not in python_path and 'SKlearn' not in python_path:
            hyperparam_spec = self.primitive.metadata.query()['primitive_code']['hyperparams']
            optimal_params = self.find_optimal_hyperparams(train=X, output=y, hyperparam_spec=hyperparam_spec,
          metric=metric_type, custom_hyperparams=custom_hyperparams)
        else:
            optimal_params = dict()

        if custom_hyperparams is not None:
            for name, value in custom_hyperparams.items():
                optimal_params[name] = value

        if 'MeanBaseline' in python_path in python_path:
            if util.invert_metric(metric_type) is True:
                return (1000000000.0, optimal_params)
            else:
                return (0.0, optimal_params)

        if y is None or 'd3m.primitives.sri' in python_path or 'bbn' in python_path:
            return (0.1, optimal_params)

        from sklearn.model_selection import KFold as KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=9001)

        splits = 5
        metric_sum = 0

        if isinstance(X, pd.DataFrame):
            Xnew = pd.DataFrame(data=X.values)
        else:
            Xnew = pd.DataFrame(data=X)

        rows = len(Xnew)
        if 'Find_projections' in python_path and 'Numeric' not in python_path:
            min_rows = (int)(rows * 0.8 * 0.5)
            if min_rows < 100:
                optimal_params['support'] = min_rows

        primitive_hyperparams = self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        prim_instance = self.primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **optimal_params))
        score = 0.0

        # Run k-fold CV and compute mean metric score
        print(Xnew.shape)
        metric_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = Xnew.iloc[train_index], Xnew.iloc[test_index]

            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train.metadata = X.metadata
            X_test.metadata = X.metadata

            prim_instance.set_training_data(inputs=X_train, outputs=y_train)

            prim_instance.fit()
            predictions = prim_instance.produce(inputs=X_test).value
            if len(predictions.columns) > 1:
                predictions = predictions.iloc[:,len(predictions.columns)-1]
            metric = self.evaluate_metric(predictions, y_test, metric_type)
            metric_scores.append(metric)
            metric_sum += metric

        score = metric_sum/splits
        print("Primitive: ", self.primitive)
        print("Metrics: ", metric_scores)
        return (score, optimal_params)

    def evaluate_metric(self, predictions, Ytest, metric):
        """
        Function to compute metric score for predicted-vs-true output.
        """
        count = len(Ytest)

        if metric is problem_pb2.ACCURACY:
            return metrics.accuracy_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION:
            return metrics.precision_score(Ytest, predictions)
        elif metric is problem_pb2.RECALL:
            return metrics.recall_score(Ytest, predictions)
        elif metric is problem_pb2.F1:
            return metrics.f1_score(Ytest, predictions)
        elif metric is problem_pb2.F1_MICRO:
            return metrics.f1_score(Ytest, predictions, average='micro')
        elif metric is problem_pb2.F1_MACRO:
            return metrics.f1_score(Ytest, predictions, average='macro')
        elif metric is problem_pb2.ROC_AUC:
            return metrics.roc_auc_score(Ytest, predictions)
        elif metric is problem_pb2.ROC_AUC_MICRO:
            return metrics.roc_auc_score(Ytest, predictions, average='micro')
        elif metric is problem_pb2.ROC_AUC_MACRO:
            return metrics.roc_auc_score(Ytest, predictions, average='macro')
        elif metric is problem_pb2.MEAN_SQUARED_ERROR:
            return metrics.mean_squared_error(Ytest, predictions)
        elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR:
            return math.sqrt(metrics.mean_squared_error(Ytest, predictions))
        elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR_AVG:
            return math.sqrt(metrics.mean_squared_error(Ytest, predictions))
        elif metric is problem_pb2.MEAN_ABSOLUTE_ERROR:
            return metrics.mean_absolute_error(Ytest, predictions)
        elif metric is problem_pb2.R_SQUARED:
            return metrics.r2_score(Ytest, predictions)
        elif metric is problem_pb2.NORMALIZED_MUTUAL_INFORMATION:
            return metrics.normalized_mutual_info_score(Ytest, predictions)
        elif metric is problem_pb2.JACCARD_SIMILARITY_SCORE:
            return metrics.jaccard_similarity_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION_AT_TOP_K:
            return 0.0
        elif metric is problem_pb2.OBJECT_DETECTION_AVERAGE_PRECISION:
            return 0.0
        else:
            return metrics.accuracy_score(Ytest, predictions)

    def optimize_primitive(self, train, output, inputs, primitive_hyperparams, optimal_params, hyperparam_types, metric_type):
        """
        Function to evaluate each input point in the hyper parameter space.
        This is called for every input sample being evaluated by the bayesian optimization package.
        Return value from this function is used to decide on function optimality.
        """
        custom_hyperparams=dict()
        for index,name in optimal_params.items():
            value = inputs[index]
            if hyperparam_types[name] is int:
                value = (int)(inputs[index]+0.5)
            custom_hyperparams[name] = value

        import random

        # Run training on 90% and testing on 10% random split of the dataset.
        seq = [i for i in range(len(train))]
        random.shuffle(seq)

        testsize = (int)(0.1 * len(train) + 0.5)
        trainindices = [seq[x] for x in range(len(train)-testsize)]
        testindices = [seq[x] for x in range(len(train)-testsize, len(train))]

        if isinstance(train, pd.DataFrame):
            Xnew = pd.DataFrame(data=train.values, columns=train.columns.values.tolist())
        else:
            Xnew = pd.DataFrame(data=train)
        Xtrain = Xnew.iloc[trainindices]
        Ytrain = output.iloc[trainindices]
        Xtest = Xnew.iloc[testindices]
        Ytest = output.iloc[testindices]

        Xtrain = d3m_dataframe(Xtrain, generate_metadata=False)
        Xtrain.metadata = train.metadata.clear(for_value=Xtrain, generate_metadata=True)
        Xtest = d3m_dataframe(Xtest, generate_metadata=False)
        Xtest.metadata = train.metadata.clear(for_value=Xtest, generate_metadata=True)
        Ytrain.metadata = output.metadata

        prim_instance = self.primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **custom_hyperparams))
        prim_instance.set_training_data(inputs=Xtrain, outputs=Ytrain)
        prim_instance.fit()

        predictions = prim_instance.produce(inputs=Xtest).value

        if len(predictions.columns) > 1:
            predictions = predictions.iloc[:,len(predictions.columns)-1]
        metric = self.evaluate_metric(predictions, Ytest, metric_type)

        if util.invert_metric(metric_type) is True:
            metric = metric * (-1)
        print('Metric: %f' %(metric))
        return metric

    def optimize_hyperparams(self, train, output, lower_bounds, upper_bounds, hyperparam_types, hyperparam_semantic_types,
     metric_type, custom_hyperparams):
        """
        Optimize primitive's hyper parameters using Bayesian Optimization package 'bo'.
        Optimization is done for the parameters with specified range(lower - upper).
        """
        domain_bounds = []
        optimal_params = {}
        index = 0
        optimal_found_params = {}

        # Create parameter ranges in domain_bounds. 
        # Map parameter names to indices in optimal_params
        for name,value in lower_bounds.items():
            if custom_hyperparams is not None and name in custom_hyperparams.keys():
                continue
            sem = hyperparam_semantic_types[name]
            if "https://metadata.datadrivendiscovery.org/types/TuningParameter" not in sem:
                continue
            lower = lower_bounds[name]
            upper = upper_bounds[name]
            if lower is None or upper is None:
                continue
            domain_bounds.append([lower,upper])
            optimal_params[index] = name
            index =index+1

        python_path = self.primitive.metadata.query()['python_path']
        if python_path == 'd3m.primitives.sri.psl.RelationalTimeseries':
            optimal_params[0] = 'period'
            index =index+1
            domain_bounds.append([1,20])

        if index == 0:
            return optimal_found_params

        primitive_hyperparams = self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        func = lambda inputs : self.optimize_primitive(train, output, inputs, primitive_hyperparams, optimal_params, hyperparam_types, metric_type)

        try:
            (curr_opt_val, curr_opt_pt) = bo.gp_call.fmax(func, domain_bounds, 50)
        except:
            print("optimize_hyperparams: ", sys.exc_info()[0])
            print(self.primitive)
            optimal_params = None

        # Map optimal parameter values found
        if optimal_params != None:
            for index,name in optimal_params.items():
                value = curr_opt_pt[index]
                if hyperparam_types[name] is int:
                    value = (int)(curr_opt_pt[index]+0.5)
                optimal_found_params[name] = value

        return optimal_found_params

    def find_optimal_hyperparams(self, train, output, hyperparam_spec, metric, custom_hyperparams):
        filter_hyperparam = lambda vl: None if vl == 'None' else vl
        hyperparam_lower_ranges = {name:filter_hyperparam(vl['lower']) for name,vl in hyperparam_spec.items() if 'lower' in vl.keys()}
        hyperparam_upper_ranges = {name:filter_hyperparam(vl['upper']) for name,vl in hyperparam_spec.items() if 'upper' in vl.keys()}
        hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in hyperparam_spec.items() if 'structural_type' in vl.keys()}
        hyperparam_semantic_types = {name:filter_hyperparam(vl['semantic_types']) for name,vl in hyperparam_spec.items() if 'semantic_types' in vl.keys()}
        optimal_hyperparams = {}
        if len(hyperparam_lower_ranges) > 0:
            optimal_hyperparams = self.optimize_hyperparams(train, output, hyperparam_lower_ranges, hyperparam_upper_ranges,
             hyperparam_types, hyperparam_semantic_types, metric, custom_hyperparams)
            print("Optimals: ", optimal_hyperparams)

        return optimal_hyperparams

