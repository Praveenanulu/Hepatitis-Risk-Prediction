"""
=========================================================
Utility Functions for Hepatitis Risk Prediction
=========================================================
Contains helper functions for:
  - Risk level classification
  - Explainable AI text generation
  - Diet plan recommendations
  - Preventive health recommendations
  - Probable hepatitis type estimation
=========================================================
"""


def risk_level(probability: float) -> tuple:
    """
    Classify risk level based on prediction probability.
    
    Returns:
        tuple: (risk_label, emoji)
    """
    if probability < 0.35:
        return "Low Risk", "🟢"
    elif probability < 0.65:
        return "Moderate Risk", "🟡"
    else:
        return "High Risk", "🔴"


def generate_explanation(
    inputs: list,
    feature_importance: dict,
    lr_coefficients: dict,
) -> str:
    """
    Generate a detailed, human-readable XAI explanation
    describing why the user received their risk prediction.
    
    Args:
        inputs: [age, alcohol, smoking, fatigue, jaundice]
        feature_importance: dict of RF feature importances
        lr_coefficients: dict of LR coefficients
    
    Returns:
        str: Paragraph-style explanation
    """
    age, alcohol, smoking, fatigue, jaundice = inputs
    factors = []

    # ── Age analysis ──
    if age > 50:
        factors.append(
            f"Your age ({age}) is a significant risk factor, as hepatitis "
            "complications increase substantially with age. Older adults often "
            "have reduced liver regeneration capacity."
        )
    elif age > 35:
        factors.append(
            f"Your age ({age}) represents a moderate risk factor. While not "
            "in the highest risk age group, liver health awareness becomes "
            "increasingly important."
        )
    else:
        factors.append(
            f"Your age ({age}) is relatively young, which is generally "
            "favorable for liver health. However, prevention remains essential."
        )

    # ── Lifestyle factors ──
    if alcohol:
        factors.append(
            "Alcohol consumption significantly increases liver damage risk "
            "and hepatitis progression. Regular alcohol intake causes direct "
            "hepatocyte damage and promotes inflammation."
        )
    if smoking:
        factors.append(
            "Smoking weakens the immune system and increases oxidative stress "
            "on liver cells, compounding hepatitis risk."
        )

    # ── Symptoms ──
    if fatigue:
        factors.append(
            "Persistent fatigue is a common early symptom of hepatitis and "
            "liver dysfunction. The liver plays a crucial role in energy "
            "metabolism."
        )
    if jaundice:
        factors.append(
            "Jaundice (yellowing of skin/eyes) is a strong indicator of "
            "liver problems, caused by elevated bilirubin levels. This "
            "symptom is heavily weighted in our model."
        )

    # ── Feature importance context ──
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    factors.append(
        f"Based on our AI model, the most influential factors are: "
        f"{top_features[0][0]} ({top_features[0][1]:.1%}), "
        f"{top_features[1][0]} ({top_features[1][1]:.1%}), and "
        f"{top_features[2][0]} ({top_features[2][1]:.1%})."
    )

    # ── LR direction analysis ──
    positive_factors = [k for k, v in lr_coefficients.items() if v > 0]
    factors.append(
        f"Logistic regression analysis confirms that "
        f"{', '.join(positive_factors)} positively correlate with "
        "hepatitis risk."
    )

    return " ".join(factors)


def diet_plan(level: str) -> str:
    """
    Generate personalized diet recommendations based on risk level.
    Includes Indian-friendly food examples.
    """
    plans = {
        "Low Risk": """
🥗 LOW RISK DIET PLAN — Maintain Your Liver Health

✅ Recommended Foods:
• Fresh fruits: Papaya, watermelon, pomegranate, amla (Indian gooseberry)
• Vegetables: Spinach (palak), bottle gourd (lauki), bitter gourd (karela)
• Whole grains: Brown rice, roti (whole wheat), oats, jowar
• Proteins: Dal (lentils), moong sprouts, paneer, grilled chicken
• Beverages: Green tea, buttermilk (chaas), coconut water

❌ Limit:
• Processed foods and excessive oil
• Sugary drinks and desserts
• Red meat consumption

💡 Tips: Eat regular, balanced meals. Stay hydrated with 8–10 glasses of water daily.
""",
        "Moderate Risk": """
🍽️ MODERATE RISK DIET PLAN — Support Liver Recovery

✅ Strongly Recommended:
• Fruits: Papaya, banana, apple, pomegranate, amla
• Vegetables: Lauki, turai, parval, palak, beetroot
• Grains: Khichdi, daliya (broken wheat), oats porridge
• Proteins: Moong dal, masoor dal, boiled eggs, steamed fish
• Liver-friendly: Turmeric milk (haldi doodh), aloe vera juice
• Beverages: Warm lemon water, herbal teas, coconut water

❌ Avoid:
• Alcohol completely
• Fried and spicy foods
• Processed and packaged foods
• Excessive salt and sugar

💡 Tips: Eat small, frequent meals. Include turmeric and garlic in cooking.
Consider liver-supporting herbs like milk thistle (consult doctor first).
""",
        "High Risk": """
🏥 HIGH RISK DIET PLAN — Intensive Liver Care

✅ Essential Foods:
• Fruits: Papaya (excellent for liver), banana, apple, pomegranate
• Vegetables: Bottle gourd juice (lauki), beetroot, carrot, pumpkin
• Easy-to-digest: Khichdi, rice kanji, moong dal soup, vegetable broth
• Proteins: Light dal water, boiled egg whites, steamed fish (small portions)
• Healing foods: Turmeric, amla, aloe vera, wheatgrass juice
• Hydration: ORS, coconut water, buttermilk, glucose water

❌ Strictly Avoid:
• ALL alcohol — absolute prohibition
• All fried, oily, and spicy foods
• Raw or undercooked foods
• Processed meats, fast food
• Heavy dairy products (cream, butter)
• Caffeinated beverages

💡 Critical Tips:
• Eat 5–6 small meals instead of 3 large ones
• Steam, boil, or bake food instead of frying
• Monitor protein intake (consult hepatologist)
• Keep a food diary and track symptoms
""",
    }
    return plans.get(level, plans["Low Risk"])


def preventive_recommendations(level: str) -> str:
    """
    Generate preventive health recommendations
    tailored to the patient's risk level.
    """
    recs = {
        "Low Risk": """
✅ PREVENTIVE RECOMMENDATIONS — Low Risk

🏃 Lifestyle:
• Maintain regular exercise (30 minutes daily)
• Ensure 7–8 hours of quality sleep
• Practice stress management (yoga, meditation)

💉 Vaccination:
• Ensure Hepatitis A and B vaccinations are up to date
• Consider booster doses if needed

🩺 Medical:
• Annual liver function tests (LFT) recommended
• Regular health check-ups every 6–12 months
• Maintain healthy BMI

🚫 Prevention:
• Practice safe hygiene and food safety
• Avoid sharing personal items (razors, toothbrushes)
• Drink clean, filtered water
""",
        "Moderate Risk": """
⚠️ PREVENTIVE RECOMMENDATIONS — Moderate Risk

🏃 Lifestyle Changes (Priority):
• Stop or significantly reduce alcohol consumption
• Quit smoking — seek professional help if needed
• Exercise moderately (walking, yoga) for 30–45 minutes daily
• Prioritize sleep and stress management

💉 Vaccination (Urgent):
• Get Hepatitis A and B vaccinations immediately if not done
• Discuss Hepatitis screening with your doctor

🩺 Medical (Important):
• Schedule liver function tests within 2 weeks
• Consult a gastroenterologist or hepatologist
• Get complete blood work done
• Consider ultrasound abdomen if recommended

🚫 Prevention (Strict):
• Avoid all hepatotoxic substances
• Practice rigorous food and water hygiene
• Avoid self-medication (especially painkillers)
• Use only prescribed medications
""",
        "High Risk": """
🚨 PREVENTIVE RECOMMENDATIONS — High Risk (URGENT)

🏥 Immediate Actions:
• Consult a hepatologist or gastroenterologist IMMEDIATELY
• Get comprehensive liver function tests (LFT, HBsAg, Anti-HCV)
• Consider liver ultrasound / fibroscan
• Get tested for all hepatitis markers (A, B, C, E)

🚫 Lifestyle (Mandatory):
• COMPLETE alcohol cessation — non-negotiable
• QUIT smoking immediately
• Avoid all OTC medications without doctor approval
• No herbal supplements without medical guidance

💉 Medical Care:
• Discuss antiviral therapy options with specialist
• Regular monitoring every 1–3 months
• Vaccination for types not yet infected
• Join a support group if needed

🏃 Supportive Care:
• Light exercise only (walking, gentle yoga)
• Complete rest during acute episodes
• Emotional and mental health support
• Family counseling about hepatitis awareness

⚠️ IMPORTANT: This is a high-risk assessment.
Please seek immediate medical attention.
""",
    }
    return recs.get(level, recs["Low Risk"])


def probable_hepatitis_type(inputs: list) -> str:
    """
    Suggest probable hepatitis type based on symptoms.
    This is a RULE-BASED estimation, not a clinical diagnosis.
    
    Args:
        inputs: [age, alcohol, smoking, fatigue, jaundice]
    
    Returns:
        str: Suggested hepatitis type(s) with disclaimer
    """
    age, alcohol, smoking, fatigue, jaundice = inputs
    suggestions = []

    if jaundice and fatigue and age < 30:
        suggestions.append(
            "Hepatitis A — Common in younger populations, often "
            "linked to contaminated food/water"
        )

    if alcohol and age > 30:
        suggestions.append(
            "Hepatitis B — Associated with chronic liver conditions, "
            "especially with alcohol use"
        )

    if alcohol and smoking and age > 40:
        suggestions.append(
            "Hepatitis C — Higher risk with chronic substance use "
            "and advancing age"
        )

    if jaundice and fatigue and not alcohol:
        suggestions.append(
            "Hepatitis E — Often waterborne, common in areas "
            "with poor sanitation"
        )

    if not suggestions:
        if jaundice or fatigue:
            suggestions.append(
                "General hepatitis risk detected — specific type "
                "requires clinical testing"
            )
        else:
            suggestions.append(
                "No specific hepatitis type strongly indicated "
                "by current symptoms"
            )

    result = "\n".join([f"• {s}" for s in suggestions])
    result += (
        "\n\n⚠️ DISCLAIMER: This is a probabilistic indication based on "
        "symptom patterns. Actual hepatitis type can ONLY be confirmed "
        "through laboratory blood tests (HBsAg, Anti-HCV, etc.). "
        "Please consult a qualified medical professional for accurate "
        "diagnosis."
    )

    return result
