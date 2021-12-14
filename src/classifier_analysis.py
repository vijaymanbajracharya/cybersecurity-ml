class ClassifierAnalysis:
    def __init__(self, algo_name, test_error, train_error, classification_error_by_label, f1_cat, f1_bin):
        self.algo_name = algo_name
        self.test_error = test_error
        self.train_error = train_error
        self.classification_error_by_label = classification_error_by_label
        self.f1_cat = f1_cat
        self.f1_bin = f1_bin
    
    def __repr__(self):
        repr = f'Algorithm Name: {self.algo_name}\n'
        repr += f'Aggregate Testing Error {self.test_error}\n'
        repr += f'Aggregate Training Error {self.train_error}\n'
        repr += f'F1 Score - Multi-Class {self.f1_cat}\n'
        repr += f'F1 Score - Binary {self.f1_bin}\n'
        repr += '----------------------------------------------\n'
        repr += '{:<15} {:<12} {:<12} {:<12}\n'.format('Label', 'Correct', 'Incorrect', 'Error')
        for key, value in self.classification_error_by_label.items():
            repr += '{:<15} {:<12} {:<12} {:<12}\n'.format(key, value[0], value[1], value[1] / (value[0] + value[1]))
        return repr