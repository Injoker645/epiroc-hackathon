# Quick Summary: Epiroc Last-Mile Delivery Optimization Hackathon

## 🎯 The Challenge

Build UX and analytics solutions to optimize last-mile deliveries by:
- **Predicting ETAs** more accurately
- **Identifying anomalies** in delivery patterns
- **Creating intuitive UX** for data exploration and root-cause analysis
- **Analyzing business impact** to prioritize optimization initiatives

**Goal**: Transform last-mile from uncertainty into customer satisfaction driver

---

## 📊 Dataset Quick Facts

- **72,966 delivery records** (Jan 2022 - Aug 2025)
- **20 features** including dates, distances, carrier info, OTD status
- **4 carrier modes**: LTL (95%), Truckload (3.3%), TL Flatbed (1.1%), TL Dry (0.5%)
- **970 unique lanes** (origin-destination pairs)
- **117 unique carriers**

### Key Performance Metrics
- **On-Time Rate**: 63.88%
- **Late Rate**: 19.18% (13,998 deliveries)
- **Early Rate**: 16.94% (12,359 deliveries)
- **Average Transit**: 2.91 days (goal: 2.79 days)

### Critical Insights
1. **LTL dominates** (95% of volume) but has 19.86% late rate
2. **Medium distances** (250-2k miles) most prone to delays
3. **Top carrier** handles 85.7% of shipments - their performance is critical
4. **Some lanes** have high volume and recurring issues

---

## 🏆 Evaluation Criteria

Jury votes with their time:
- **3 points**: Want a 20-minute follow-up meeting
- **1 point**: Interesting, will provide feedback
- **0 points**: Not for me

**Focus**: Real business interest and practical value

---

## 📅 Timeline

**Hack Day**: December 16th
- 09:30 - Hacking starts
- 17:30 - Team Presentations (4 minutes each)
- 19:00 - Celebration dinner

**Preparation**:
- Dec 3: Business Value Seminar
- Dec 4: Bluetext Prototyping Platform Lab
- Dec 5: GitHub Copilot Training

---

## 🛠️ Resources Provided

- **GitHub Copilot Business** license
- **$50 OpenRouter AI credits** (watch token usage!)
- **Virtual dev machines** (Bluetext platform)
- **Discord support** channels

---

## 💡 Solution Focus Areas

### 1. Lead Time Prediction
- Build models to predict transit days accurately
- Account for carrier mode, distance, lane, temporal patterns
- Improve ETA accuracy beyond current 63.88% on-time rate

### 2. Anomaly Detection
- Identify unusual patterns (negative transit days, extreme delays)
- Flag carrier-specific issues
- Detect lane-specific problems

### 3. UX for Root-Cause Analysis
- Interactive dashboards for exploring delivery data
- Filter by carrier, distance, lane, time period
- Visualize patterns and trends
- Enable drilling down to identify root causes

### 4. Business Impact Analysis
- Quantify cost of delays
- Prioritize optimization initiatives
- Show ROI of improvements
- Guide resource allocation

---

## 🎯 Winning Strategy

1. **Scope it down**: Build one compelling feature, not everything
2. **Show business value**: Address mission-critical activity
3. **Make it interactive**: Demo user interaction, not just charts
4. **Demonstrate impact**: Show how it improves decision-making
5. **Present future vision**: Explain production potential

---

## 📋 SLA Context

- **Gross OTD Target**: 95% (includes all delays)
- **Controllable OTD Target**: 98% (carrier-controlled only)
- Current performance: 63.88% on-time
- **Opportunity**: Improve by ~31 percentage points to reach target

---

## 🚀 Next Steps

1. ✅ **EDA Complete** - Understand data patterns
2. ⏭️ **Data Cleaning** - Handle anomalies (negative transit days, outliers)
3. ⏭️ **Feature Engineering** - Create predictive features
4. ⏭️ **Model Development** - Build prediction and anomaly detection models
5. ⏭️ **Dashboard Development** - Create interactive UX
6. ⏭️ **Business Impact Analysis** - Quantify improvements

---

## 📁 Files Created

- `EDA_SUMMARY.md` - Detailed EDA findings
- `COMPETITION_DETAILS.md` - Full competition information
- `eda_analysis.py` - EDA script
- `QUICK_SUMMARY.md` - This file


