import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# file_path = r"C:\Users\robbi\OneDrive\Documents\TDA Workk\1000 iterations of fundamental class on 2sphere.xlsx"
# n_lifetime_df = pd.read_excel(file_path)
#
# # Now you can work with the DataFrame 'df'
#
#
# # n_lifetime_df = pd.DataFrame(data = n_lifetime_df, columns = ['Sample size', 'Lifetime of Fundamental Class'])
#
# print(n_lifetime_df)
#
#
# n_lifetime_df.boxplot(by='Sample_Size', column =['Lifetime_of_Fundamental_class'],
#                                 grid=False)
#
# # plt.axhline(y = np.sqrt(2), color='r', linestyle = '-')
#
# plt.show()


# Monte carlo on S1 with Std = 0.01


#################################################
# Data of std = 0.01
# file_path = r"C:\Users\robbi\OneDrive\Documents\TDA Workk\S1 samples of Lifetime with Std of 0.01.xlsx"
#
# n_lifetime_df = pd.read_excel(file_path)
#
# n_lifetime_df = pd.DataFrame(data = n_lifetime_df, columns = ['Sample_Size', 'Lifetime'])
#
#
# plt.scatter(n_lifetime_df[['Sample_Size']], n_lifetime_df[['Lifetime']])
#
# plt.show()
#
#############################################################

##############################################################
# Data of Std = 0
file_path = r"C:\Users\robbi\OneDrive\Documents\TDA Workk\Monte Carlo on S1 with Std of 0.xlsx"

n_lifetime_df = pd.read_excel(file_path)

n_lifetime_df = pd.DataFrame(data = n_lifetime_df, columns = ['Sample_Size', 'Lifetime'])


plt.scatter(n_lifetime_df[['Sample_Size']], n_lifetime_df[['Lifetime']])

plt.show()


###############################################################






# n_lifetime_df['Bins'] = pd.cut(n_lifetime_df['Sample_Size'], bins=10)
#
# n_lifetime_df.boxplot(column='Lifetime', by='Bins')
# plt.xlabel('Bins')
# plt.ylabel('Lifetime')
# plt.title('Boxplot of Lifetime by Bins of Sample Sizes')
#
# # Slant the x-axis labels at 45 degrees
# plt.xticks(rotation=30)
#
# plt.show()







# file_path = r"C:\Users\robbi\OneDrive\Documents\TDA Workk\Dataframe of sample sizes and lifetimes of S1 persistance lifetime.xlsx"
#
# n_lifetime_df = pd.read_excel(file_path)
#
# plt.scatter(n_lifetime_df[['Sample_Size']], n_lifetime_df[['Lifetime']])
#
# plt.show()

# Now you can work with the DataFrame 'df'


# n_lifetime_df = pd.DataFrame(data = n_lifetime_df, columns = ['Sample_Size', 'Lifetime'])
#
# # print(n_lifetime_df)
#
#
# n_lifetime_df.boxplot(by='Sample_Size', column =['Lifetime'],
#                                 grid=False)
#
# plt.axhline(y = np.sqrt(2), color='r', linestyle = '-')
#
# plt.show()
#
# plt.scatter(n_lifetime_df[:,0], n_lifetime_df[:,1])
# plt.show()
#
#
#
#
#
#
#
# n_lifetime_df = pd.DataFrame(data = n_lifetime_df, columns = ['Sample_Size', 'Lifetime'])
#
# n_lifetime_df['Bins'] = pd.cut(n_lifetime_df['Sample_Size'], bins=10)

# # n_lifetime_df.boxplot(column='Lifetime', by='Bins')
# # plt.xlabel('Bins')
# # plt.ylabel('Lifetime')
# # plt.title('Boxplot of Lifetime by Bins of Sample Sizes')
# #
# # # Slant the x-axis labels at 45 degrees
# # plt.xticks(rotation=30)
# #
# # plt.show()
#
