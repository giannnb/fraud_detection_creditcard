from utils import Utils
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

if __name__ == "__main__":

    utils = Utils()

    dataset = utils.load_csv('data/creditcard.csv')

    X, y = utils.variables(dataset, 'Class', 'Class')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = RobustScaler()
    
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test[['Amount', 'Time']] = scaler.fit_transform(X_test[['Amount', 'Time']])

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_smote.fit(X_train_resampled, y_train_resampled)

    y_pred_smote = rf_model_smote.predict(X_test)

    print("--- REPORTE FINAL CON SMOTE ---")
    print(classification_report(y_test, y_pred_smote))

    utils.model_export(rf_model_smote)
