class TreinamentoAvaliacaoKfold:
  import sys
  import numpy as np

  from sklearn import tree
  from sklearn.naive_bayes import GaussianNB
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.neural_network import MLPClassifier
  from sklearn import svm
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import GradientBoostingClassifier
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.gaussian_process import GaussianProcessClassifier
  from sklearn.model_selection import cross_val_score
  #from sklearn.model_selection import StratifiedKFold
  from sklearn.model_selection import KFold

  def ler_dataset(self, arquivo_dataset):
    f = open(arquivo_dataset, "r")
    dataset_array=[]
    label_array=[]
    for x in f:
        data = []
        label= 0
        for dado in x.split(';'):
            if "#" in dado:
                data.append(int(dado.split('#')[0]))
                label=int(dado.split('#')[1].split('\n')[0])
            else:
                data.append(int(dado))
        dataset_array.append(data)
        label_array.append(label)  
    f.close()
    #print(dataset_array)
    #print(label_array)
    return dataset_array,label_array

  if __name__ == '__main__':
    if len(sys.argv) != 3:
       print("Erro na passagem dos argumentos")
       print("Correto:")
       print("        python3 <nome_arquivo_dataset> <valor K de folds>")
    else:
       #constantes
       arquivo_dataset   =sys.argv[1]
       k		 =sys.argv[2]
       from TreinamentoAvaliacaoKfold import TreinamentoAvaliacaoKfold

       print("***************************************")
       print("Valor do K folds: {}".format(k))	

       ta = TreinamentoAvaliacaoKfold()
       X_train, Y_train = ta.ler_dataset(arquivo_dataset)
       #data_teste, label_teste = ta.ler_dataset(arquivo_datateste)

       modelos = []
       modelos.append(('rf',RandomForestClassifier(n_estimators=300)))
       modelos.append(('mlp',MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1, max_iter=300)))
       modelos.append(('gbc',GradientBoostingClassifier(n_estimators=300)))
       modelos.append(('abc',AdaBoostClassifier(n_estimators=300)))
       modelos.append(('gpc',GaussianProcessClassifier(max_iter_predict=300)))
       modelos.append(('svm',svm.SVC(max_iter=300)))
       modelos.append(('nb',GaussianNB()))
       modelos.append(('knn',KNeighborsClassifier(n_neighbors=3)))
       modelos.append(('tree',tree.DecisionTreeClassifier(max_depth=300)))

       resultados = []
       nomes = []
       for nome, modelo in modelos:
          #kfold = StratifiedKFold(n_splits=8, random_state=1)
          kfold  = KFold(n_splits=8, shuffle=True)
          cv_results = cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring='accuracy')
          #cv_results = cross_val_score(modelo, X_train, Y_train, cv=kfold, scoring='f1_macro')
          resultados.append(cv_results)
          nomes.append(nome)
          print('%s: %f (%f)' % (nome, cv_results.mean(), cv_results.std()))

