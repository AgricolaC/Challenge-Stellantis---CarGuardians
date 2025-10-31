from imblearn.combine import SMOTEENN
from sklearn.preprocessing import MinMaxScaler

def scale_and_balance(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_scaled, y)
    print(f"After balancing: {sum(y_res==0)} negatives, {sum(y_res==1)} positives.")
    return X_res, y_res,scaler
