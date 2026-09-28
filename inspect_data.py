import pandas as pd

df = pd.read_csv('social_media_brand_analytics.csv')

print('=== SHAPE ===')
print(df.shape)

print('\n=== COLUMNS & DTYPES ===')
print(df.dtypes)

print('\n=== MISSING VALUES ===')
print(df.isnull().sum())

print('\n=== DUPLICATE ROWS ===')
print('Duplicate rows:', df.duplicated().sum())
if 'Post_ID' in df.columns:
    print('Duplicate Post_IDs:', df['Post_ID'].duplicated().sum())

print('\n=== TIMESTAMP RANGE ===')
ts_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
for c in ts_col:
    parsed = pd.to_datetime(df[c], errors='coerce')
    print(f'{c}: {parsed.min()} to {parsed.max()}')

print('\n=== UNIQUE PLATFORMS ===')
plat_col = [c for c in df.columns if 'platform' in c.lower()]
for c in plat_col:
    print(df[c].value_counts())

print('\n=== SENTIMENT DISTRIBUTION ===')
sent_col = [c for c in df.columns if 'sentiment' in c.lower()]
for c in sent_col:
    print(df[c].value_counts())

print('\n=== ENGAGEMENT STATS ===')
eng_col = [c for c in df.columns if any(k in c.lower() for k in ['engagement','like','share','comment','retweet','view'])]
for c in eng_col:
    print(f'{c}: min={df[c].min()}, max={df[c].max()}, mean={df[c].mean():.2f}, median={df[c].median()}')

print('\n=== UNIQUE POSTS ===')
if 'Post_ID' in df.columns:
    print('Unique Post_IDs:', df['Post_ID'].nunique())

print('\n=== SAMPLE 5 ROWS ===')
print(df.sample(5, random_state=42).to_string())
