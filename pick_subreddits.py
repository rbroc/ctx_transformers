import pandas as pd 
df = pd.read_csv('https://frontpagemetrics.com/files/2020-11-13.csv', encoding='latin')
df = df[df.real_name.str.contains(r'[P|p]orn') == False]
top_1000 = df.sort_values('subs', ascending=False).reset_index()[['real_name', 'subs']].iloc[:1000]
top_1000.to_csv('../misc/top_1000_subreddits.txt', sep='\t')