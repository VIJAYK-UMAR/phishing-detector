import re
import socket
import urllib.parse
import ipaddress
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.utils import resample
import joblib

# --------------------------------------
# 1. FEATURE EXTRACTION FROM A URL (30)
# --------------------------------------

def extract_features(url: str) -> list:
    """
    Given a URL string, compute the 30 features in the order used
    by the UCI “Phishing Websites” dataset. Each feature returns –1, 0, or 1.

    Feature Index (1-based) and Description:
    1.  having_IP_Address
    2.  URL_Length
    3.  Shortening_Service
    4.  having_At_Symbol
    5.  double_slash_redirecting
    6.  Prefix_Suffix (hyphen in domain)
    7.  having_Sub_Domain
    8.  SSLfinal_State
    9.  Domain_registeration_length       (WHOIS lookup, placeholder)
    10. favicon                          (placeholder)
    11. port                             (placeholder)
    12. HTTPS_token
    13. Request_URL
    14. URL_of_Anchor
    15. Links_in_tags
    16. SFH (Server Form Handler)
    17. Submitting_to_email
    18. Abnormal_URL
    19. Redirect
    20. on_mouseover
    21. RightClick
    22. popUpWidnow
    23. IFrame
    24. age_of_domain                    (WHOIS lookup, placeholder)
    25. DNSRecord                        (DNS lookup, placeholder)
    26. web_traffic                      (external lookup, placeholder)
    27. Page_Rank                        (external lookup, placeholder)
    28. Google_Index                     (external lookup, placeholder)
    29. Links_pointing_to_page           (external lookup, placeholder)
    30. Statistical_report               (external lookup, placeholder)
    """

    features = []

    # Normalize URL
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    path = (parsed.path or "").lower()
    full_url = url.lower()

    # 1. having_IP_Address
    #    – If the domain is an IP address → phishing (–1)
    #    – Otherwise → legitimate (1)
    try:
        ipaddress.ip_address(domain)
        features.append(-1)
    except Exception:
        features.append(1)

    # 2. URL_Length
    #    – < 54 chars → 1 (legit)
    #    – 54 to 75 chars → 0 (suspicious)
    #    – > 75 chars → –1 (phishing)
    url_len = len(full_url)
    if url_len < 54:
        features.append(1)
    elif 54 <= url_len <= 75:
        features.append(0)
    else:
        features.append(-1)

    # 3. Shortening_Service
    #    – If URL uses a known “tinyurl-style” service (bit.ly, goo.gl, etc.) → –1
    #    – Otherwise → 1
    shortening_patterns = re.compile(
        r"(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly"
        r"|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com"
        r"|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac"
        r"|su\.pr|twurl\.nl|sn\.im|short\.to|BudURL\.com|ping\.fm"
        r"|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us"
        r"|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com"
        r"|om\.ly|to\.ly|bit\.do|t\.ly)"
    )
    if shortening_patterns.search(full_url):
        features.append(-1)
    else:
        features.append(1)

    # 4. having_At_Symbol
    #    – If “@” in URL → –1
    #    – Otherwise → 1
    if "@" in full_url:
        features.append(-1)
    else:
        features.append(1)

    # 5. double_slash_redirecting
    #    – If “//” appears after the protocol (other than the “http://” or “https://” part) → –1
    #      e.g. http://example.com//evil
    #    – Otherwise → 1
    if full_url.count("//") > 1:
        features.append(-1)
    else:
        features.append(1)

    # 6. Prefix_Suffix (hyphen in domain)
    #    – If “-” in domain → –1
    #    – Otherwise → 1
    if "-" in domain:
        features.append(-1)
    else:
        features.append(1)

    # 7. having_Sub_Domain
    #    – Count subdomains by splitting domain on “.”
    #    – > 2 subdomains → –1, exactly 2 subdomains → 0, <= 1 → 1
    domain_parts = domain.split(".")
    if len(domain_parts) <= 2:
        features.append(1)
    elif len(domain_parts) == 3:
        features.append(0)
    else:
        features.append(-1)

    # 8. SSLfinal_State
    #    We look at:
    #      1) URL starts with “https://”  (good)
    #      2) Certificate validity and issuing CA (requires external check). 
    #    For simplicity, check only “https” vs “http”:
    #    – If URL begins with “https://” → 1
    #    – Else → –1
    if parsed.scheme == "https":
        features.append(1)
    else:
        features.append(-1)

    # 9. Domain_registeration_length   (placeholder)
    features.append(0)

    # 10. favicon                       (placeholder)
    #     For now, default to 0 (suspicious).
    features.append(0)

    # 11. port                          (placeholder)
    #     – If URL explicitly specifies an unusual port (e.g. http://example.com:8080) → –1
    #     – If default port (80 for http, 443 for https) → 1
    #     Here we’ll parse and check:
    try:
        _, port = domain.split(":")
        port = int(port)
        if (parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443):
            features.append(1)
        else:
            features.append(-1)
    except ValueError:
        # No explicit port → 1
        features.append(1)

    # 12. HTTPS_token
    #     – If “https” appears in domain part (not at the beginning of URL) → –1
    #     – Otherwise → 1
    if re.search(r"https", domain):
        features.append(-1)
    else:
        features.append(1)

    # 13. Request_URL
    #     – Ratio of external objects loaded to total objects. 
    #       If > 0.61 → –1, if 0.31–0.61 → 0, else → 1.
    #     Here we fetch HTML and count <img> and <audio> tags for external domains:
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        total_urls = len(soup.find_all(src=True))
        if total_urls == 0:
            features.append(1)
        else:
            external = 0
            for tag in soup.find_all(src=True):
                src = tag["src"]
                # If “src” starts with “http” and domain not in it → external
                if src.startswith("http") and (urllib.parse.urlparse(src).netloc != domain):
                    external += 1
            ratio = external / total_urls
            if ratio < 0.31:
                features.append(1)
            elif 0.31 <= ratio <= 0.61:
                features.append(0)
            else:
                features.append(-1)
    except Exception:
        # On any error (timeout, bad HTML), mark as 0 (suspicious)
        features.append(0)

    # 14. URL_of_Anchor
    #     – Count <a> tags and see how many have “href” pointing outside domain.
    #       If > 0.67 → –1, if 0.33–0.67 → 0, else → 1.
    try:
        anchor_tags = soup.find_all("a", href=True)
        if len(anchor_tags) == 0:
            features.append(1)
        else:
            ext = 0
            for a in anchor_tags:
                href = a["href"]
                if href.startswith("http"):
                    if urllib.parse.urlparse(href).netloc != domain:
                        ext += 1
            ratio = ext / len(anchor_tags)
            if ratio < 0.31:
                features.append(1)
            elif 0.31 <= ratio <= 0.67:
                features.append(0)
            else:
                features.append(-1)
    except Exception:
        features.append(0)

    # 15. Links_in_tags
    #     – Look at <link> and <meta> tags: count external vs total. 
    #       If > 0.5 → –1, if 0.17–0.5 → 0, else → 1.
    try:
        link_tags = soup.find_all("link", href=True)
        meta_tags = soup.find_all("meta", content=True)
        total_tags = len(link_tags) + len(meta_tags)
        if total_tags == 0:
            features.append(1)
        else:
            ext = 0
            for tag in link_tags:
                href = tag["href"]
                if href.startswith("http") and (urllib.parse.urlparse(href).netloc != domain):
                    ext += 1
            for tag in meta_tags:
                content = tag.get("content", "")
                if content.startswith("http") and (urllib.parse.urlparse(content).netloc != domain):
                    ext += 1
            ratio = ext / total_tags
            if ratio < 0.17:
                features.append(1)
            elif 0.17 <= ratio <= 0.5:
                features.append(0)
            else:
                features.append(-1)
    except Exception:
        features.append(0)

    # 16. SFH (Server Form Handler)
    #     – If <form> action is “about:blank” or missing → –1
    #     – If external domain form action → 0
    #     – If same domain → 1
    try:
        forms = soup.find_all("form", action=True)
        if not forms:
            features.append(-1)
        else:
            status = 1
            for f in forms:
                action = f["action"]
                if action == "" or action.lower() == "about:blank":
                    status = -1
                    break
                elif action.startswith("http") and (urllib.parse.urlparse(action).netloc != domain):
                    status = 0
            features.append(status)
    except Exception:
        features.append(0)

    # 17. Submitting_to_email
    #     – If “mailto:” found in form action → –1, else → 1
    if re.search(r"mailto:", full_url):
        features.append(-1)
    else:
        features.append(1)

    # 18. Abnormal_URL
    #     – If domain in WHOIS record doesn’t match URL domain → –1  
    #       (requires WHOIS; placeholder = 0)
    features.append(0)

    # 19. Redirect
    #     – If more than one “//” (beyond initial “http://”) → –1 else → 1
    if full_url.count("//") > 1:
        features.append(-1)
    else:
        features.append(1)

    # 20. on_mouseover
    #     – If “onmouseover=” present in HTML → –1 else → 1
    try:
        if re.search(r"onmouseover\s*=", resp.text):
            features.append(-1)
        else:
            features.append(1)
    except Exception:
        features.append(0)

    # 21. RightClick
    #     – If “event.button==2” or “contextmenu” in JS → –1 else → 1
    try:
        if re.search(r"event\.button\s*==\s*2", resp.text) or re.search(r"contextmenu", resp.text):
            features.append(-1)
        else:
            features.append(1)
    except Exception:
        features.append(0)

    # 22. popUpWidnow
    #     – If “window.open(” in JS → –1 else → 1
    try:
        if re.search(r"window\.open\(", resp.text):
            features.append(-1)
        else:
            features.append(1)
    except Exception:
        features.append(0)

    # 23. IFrame
    #     – If “<iframe” found in HTML → –1 else → 1
    try:
        if soup.find("iframe"):
            features.append(-1)
        else:
            features.append(1)
    except Exception:
        features.append(0)

    # 24. age_of_domain                 (placeholder: WHOIS needed)
    features.append(0)

    # 25. DNSRecord                     (placeholder: DNS lookup needed)
    features.append(0)

    # 26. web_traffic                   (placeholder: external “Alexa” or similar)
    features.append(0)

    # 27. Page_Rank                     (placeholder: external API)
    features.append(0)

    # 28. Google_Index                  (placeholder: external API)
    features.append(0)

    # 29. Links_pointing_to_page        (placeholder: external API)
    features.append(0)

    # 30. Statistical_report            (placeholder: historic data)
    features.append(0)

    # Finally, return the 30-element list (values –1, 0, or 1)
    return features


# ---------------------------------------------------
# 2. LOAD, EXPLORE, AND (OPTIONALLY) BALANCE THE DATA
# ---------------------------------------------------

# Load the phishing dataset CSV (with 30 features + “class”)
df = pd.read_csv("phishing.csv")

# Drop any non‐informative column (e.g. “Index” if present)
if "Index" in df.columns:
    df = df.drop(columns=["Index"])

# Display basic info
# print("----- DATA HEAD -----")
# print(df.head(), "\n")
# print("----- DATA INFO -----")
# print(df.info(), "\n")
# print("----- CLASS DISTRIBUTION -----")
# print(df["class"].value_counts(), "\n")

# Check for imbalance, and upsample minority if needed
count_legit = df[df["class"] == 1].shape[0]
count_phish = df[df["class"] == -1].shape[0]

if count_legit != count_phish:
    # print(f"Balancing dataset: Legit={count_legit}, Phishing={count_phish}")
    df_legit = df[df["class"] == 1]
    df_phish = df[df["class"] == -1]

    if count_legit > count_phish:
        df_phish_upsampled = resample(
            df_phish,
            replace=True,
            n_samples=count_legit,
            random_state=42,
        )
        df_balanced = pd.concat([df_legit, df_phish_upsampled])
    else:
        df_legit_upsampled = resample(
            df_legit,
            replace=True,
            n_samples=count_phish,
            random_state=42,
        )
        df_balanced = pd.concat([df_legit_upsampled, df_phish])

    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    # print("After balancing:", df["class"].value_counts(), "\n")


# ---------------------------------------------------
# 3. SPLIT FEATURES & TARGET, THEN TRAIN/TEST SPLIT
# ---------------------------------------------------

X = df.drop(columns=["class"])  # 30 feature columns
y = df["class"]                  # –1 = phishing, 1 = legitimate

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ---------------------------------------------------
# 4. MODEL COMPARISON: LOGISTIC, RANDOM FOREST, SVM
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
}

# print("----- MODEL COMPARISON -----")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # print(f"{name:20s} | Test Accuracy: {acc:.4f}")
# print()

# ---------------------------------------------------
# 5. CHOOSE FINAL MODEL (e.g. LOGISTIC) & EVALUATE
# ---------------------------------------------------

final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test, y_pred, target_names=["Phishing", "Legit"]
)

# print("----- FINAL MODEL EVALUATION -----")
# print(f"Accuracy: {accuracy:.4f}\n")
# print("Classification Report:\n", report)

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test, cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression")
# plt.show()

# ---------------------------------------------------
# 6. FEATURE IMPORTANCE (COEFFICIENTS FOR LOGISTIC)
# ---------------------------------------------------

# print("----- FEATURE IMPORTANCE (Logistic Regression) -----")
coef = final_model.coef_[0]
feature_importance = sorted(
    zip(X.columns, coef), key=lambda x: abs(x[1]), reverse=True
)
# for feature, weight in feature_importance:
#     print(f"{feature:30s}: {weight:.4f}")
# print()

# ---------------------------------------------------
# 7. SAVE THE TRAINED MODEL TO DISK
# ---------------------------------------------------

# joblib.dump(final_model, "phishing_model.pkl")
# print("Trained model saved as phishing_model.pkl\n")

# ---------------------------------------------------
# 8. PREDICT ON A NEW URL AT RUNTIME
# ---------------------------------------------------

def predict_url(url: str):
    """
    Given a URL string, extract features → load saved model → predict.
    """
    feats = extract_features(url)
    # print(feats)# 30‐dimensional feature vector
    feats_df = pd.DataFrame([feats], columns=X.columns)  # Make same columns
    model = joblib.load("phishing_model.pkl")            # Load saved model
    pred = model.predict(feats_df)[0]  
    if pred == 1:
        print(f"✅ PREDICTED as LEGITIMATE.")
    else:
        print(f"🚨 PREDICTED as PHISHING.")


# Example usage:
if __name__ == "__main__":
    test_url_1 = input("ENTER URL : ")
    print("----- TEST PREDICTIONS -----")
    print()
    predict_url(test_url_1)
