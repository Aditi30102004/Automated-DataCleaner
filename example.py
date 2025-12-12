import pandas as pd
from datacleaner_aditi import Cleaner

df = pd.DataFrame({"A": [1, 2, None], "B": ["x", None, "y"]})
cleaner = Cleaner()
df_clean = cleaner.fit_transform(df)
print(df_clean)
