
import pandas as pd
import matplotlib.pyplot as plt


def exhaustive_test(folder_name, dtype = 'mat'):
  metric_types = ['ap', 'prc', 'roc', 'time']
  titles = ['Average Precision', 'PRC', 'ROC-AUC', 'Time']
  
  for i in range(4):
    filename = f'{folder_name}/{metric_types[i]}_{dtype}.csv'
    df = pd.read_csv(filename)
    df = df[['COD', 'BAZUOD']]
    win = (df['BAZUOD'].mean() - df['COD'].mean()) / df['BAZUOD'].mean() * 100
    print(f'------Average Performance ({titles[i]}) ------')
    print('    COPOD: ', df['COD'].mean())
    print('    BAZUOD:', df['BAZUOD'].mean())
    if i == 3:
      print(f'    BAZUOD takes {win:.2f}% longer than COPOD')
    else:
      print(f'    BAZUOD perform {win:.2f}% better than COPOD')


if __name__ == "__main__":
  exhaustive_test('trial2', 'mat')
  exit()

  metric_types = ['ap', 'prc', 'roc', 'time']
  titles = ['Average Precision', 'PRC', 'ROC-AUC', 'Time']
  
  # Change these properties to plot with different setting
  folder = 'trial5'
  selection = 2
  dtype = 'arff'

  filename = f'{folder}/{metric_types[selection]}_{dtype}.csv'

  df = pd.read_csv(filename)
  df = df[['COD', 'BAZUOD']]

  win = (df['BAZUOD'].mean() - df['COD'].mean()) / df['BAZUOD'].mean() * 100
  print(f'------Average Performance ({titles[selection]}) ------')
  print('COPOD: ', df['COD'].mean())
  print('BAZUOD:', df['BAZUOD'].mean())
  if selection == 3:
    print(f'    BAZUOD takes {win:.2f}% longer than COPOD')
  else:
    print(f'    BAZUOD perform {win:.2f}% better than COPOD')

  xVals = range(1, len(df) + 1)
  plt.plot(xVals, df['COD'], label = 'COPOD')
  plt.plot(xVals, df['BAZUOD'], label = 'BAZUOD')
  plt.title(titles[selection])
  plt.xlabel('Test Dataset ID')
  plt.ylabel(titles[selection])
  plt.legend()
  plt.show()
