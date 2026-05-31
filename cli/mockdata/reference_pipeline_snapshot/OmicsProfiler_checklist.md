# OmicsProfiler Background Checklist: Kidney Tissue Cell Composition

**Dataset Reference:** Sample s255
**Clinical Context:** Diabetic Kidney Disease (DKD) Progression
**Primary Hypothesis:** DKD progression is characterized by a quantifiable loss of functional Proximal Tubules and a concurrent upregulation of Fibrosis Markers (CollagenIV, aSMA), with a specific focus on identifying hidden damage in late-stage tubules.

Use the following cell composition checklist to guide the automated interpretation agent in analyzing spatial ROIs across key kidney tissue structures.

---

### 1. Proximal Tubules
*Focus: Assess for loss of functional markers and thickening of the basement membrane indicative of DKD progression.*
* **Enriched:**
  * Functional Proximal Tubular cells (CD183++/CD227-) *Note: Monitor for quantifiable loss of CD183 expression in late-stage DKD.*
  * Fibrosis/Basement Membrane markers (CollagenIV+) *Note: May be upregulated around damaged tubules.*
* **Sparse:**
  * Distal Tubular cells (CD227-)
  * Podocytes (Nestin-)
  * Endothelial cells (CD31-)

### 2. Distal Tubules
*Focus: Differentiate from proximal tubules to establish accurate tubular composition and structural integrity.*
* **Enriched:**
  * Distal Tubular cells (CD227++/CD183-)
* **Sparse:**
  * Proximal Tubular cells (CD183-)
  * Podocytes (Nestin-)
  * Myeloid/Immune cells (CD68-, CD11b-) unless severe interstitial inflammation is encroaching.

### 3. Glomeruli
*Focus: Identify structural integrity of the filtration barrier and signs of glomerulosclerosis (fibrosis).*
* **Enriched:**
  * Podocytes (Nestin++)
  * Glomerular Endothelial cells (CD31+)
  * Mesangial/Profibrotic cells (aSMA+, CollagenIV+) *Note: Significant upregulation expected in late-stage DKD.*
* **Sparse:**
  * Tubular cells (CD183-, CD227-)
  * General structural leukocytes (CD45-), though infiltrating immune cells may be present in severe disease.

### 4. Blood Vessel
*Focus: Distinguish vascular structures from fibrotic interstitium and evaluate vascular smooth muscle.*
* **Enriched:**
  * Vascular Endothelial cells (CD31++)
  * Vascular Smooth Muscle cells (aSMA++)
* **Sparse:**
  * Tubular cells (CD183-, CD227-)
  * Podocytes (Nestin-)
  * Interstitial Macrophages (CD68-) *within the vessel wall.*

### 5. Interstitium
*Focus: Quantify the upregulation of fibrosis markers and characterize the inflammatory immune infiltrate driving DKD damage.*
* **Enriched:**
  * Myofibroblasts / Fibroblasts (aSMA++, CollagenIV++) *Note: Key indicators of interstitial fibrosis.*
  * Pan-Leukocytes (CD45++)
  * Macrophages / Myeloid cells (CD68++, CD11b++)
  * Th17 / Specialized Immune Subsets (CD196+)
* **Sparse:**
  * Functional Tubular cells (CD183-, CD227-) *Note: Presence in the interstitium indicates severe tubular atrophy/destruction.*
  * Podocytes (Nestin-)