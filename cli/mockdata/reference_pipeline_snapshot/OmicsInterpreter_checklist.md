BACKGROUND KNOWLEDGE: CROSS-MODAL ALIGNMENT & CONFLICT RESOLUTION

**Primary Hypothesis Context:** DKD progression is characterized by a quantifiable loss of functional Proximal Tubules (PT) and a concurrent upregulation of Fibrosis Markers (Collagen IV, aSMA), with a specific focus on identifying hidden damage in late-stage tubules.

When analyzing Sample s255, the automated interpretation agent must apply the following cross-modal alignment and conflict resolution rules to reconcile visual structural data with spatial omics data (cellular composition and spatially variable genes).

### 1. Glomeruli (High Visual Reliability)
Glomeruli possess highly distinct morphological features (capillary tufts, Bowman's capsule, Bowman's space) that are robustly identifiable in visual modalities.
*   **Conflict Rule (Trust Visual):** If spatial omics indicates a high proportion of non-glomerular cell types (e.g., Proximal Tubule cells) within a clearly defined glomerular structure, trust the visual morphology. This conflict is typically caused by transcript/signal spillover or segmentation artifacts from adjacent tightly packed tubules.
*   **Alignment (Omics Confirmation):** Visual identification of a glomerulus should be corroborated by omics signals for resident glomerular cells: Podocytes (e.g., *NPHS1*, *NPHS2*, *WT1*), Glomerular Endothelial Cells (*PECAM1*/*CD31*), and Mesangial Cells. A lack of these markers in a visually intact glomerulus may indicate severe glomerular sclerosis or localized omics dropout.

### 2. Blood Vessels (Structure vs. Cell Type Disambiguation)
In the context of DKD, aSMA (alpha-smooth muscle actin) is highly upregulated. It serves as a marker for both vascular smooth muscle cells (normal/thickened vessels) and myofibroblasts (interstitial fibrosis).
*   **Conflict Rule (Disambiguate aSMA+):** To distinguish blood vessels from fibrotic interstitial tissue, rely on visual structural cues. If an aSMA+ region lacks a defined lumen or is scattered irregularly between tubules, it represents interstitial myofibroblasts (fibrosis). If the aSMA+ signal concentrically surrounds a distinct lumen, it is a blood vessel.
*   **Alignment (Structural Confirmation):** For an aSMA+ region to be classified as a blood vessel, the visual modality must confirm a tubular/circular structure with a clear lumen. Additionally, omics or visual data should ideally confirm an inner endothelial lining (e.g., *CD31*/*PECAM1*+ or *ERG*+) adjacent to the aSMA signal.

### 3. Interstitium vs. Proximal Tubules (High Omics Reliability)
In late-stage DKD, Proximal Tubules undergo severe atrophy, basement membranes thicken, and the interstitium expands due to fibrosis (Collagen IV, aSMA). This causes visual boundaries between collapsed tubules and the fibrotic interstitium to blur.
*   **Conflict Rule (Trust Omics):** When visual boundaries are ambiguous, fragmented, or suggest "hidden damage" (e.g., a tubule that looks structurally compromised or visually blends into the surrounding matrix), trust the omics data to define the compartment.
*   **Alignment (Cell Composition Pattern):** True interstitium will be confirmed by a cellular composition dominated by Fibroblasts/Myofibroblasts (aSMA+, high Collagen IV expression), Macrophages, and other immune infiltrates. Conversely, if the region retains a high density of PT-specific transcripts despite visual distortion, it should be classified as a damaged/atrophic Proximal Tubule rather than pure interstitium.

### 4. Proximal Tubules vs. Distal Tubules (High Omics Reliability)
Visually distinguishing Proximal Tubules from Distal Tubules can be challenging, particularly when structural integrity is compromised by advanced DKD (e.g., loss of the classic PT brush border). Omics provides high-fidelity differentiation.
*   **Conflict Rule (Differentiate via Omics):** Do not rely solely on visual luminal diameter or epithelial thickness to distinguish PT from DT in late-stage DKD. Always defer to the omics biomarker signature to classify the tubule type.
*   **Alignment (Key Biomarker Differentiators):**
    *   **Proximal Tubules (PT):** Must exhibit a **CD183++ / CD227-** signature. A decline in functional PT markers alongside this signature indicates the "quantifiable loss of function" targeted by the hypothesis.
    *   **Distal Tubules (DT):** Must exhibit a **CD227+** signature.
    *(Note: Ensure strict adherence to these specific marker profiles to prevent misclassification of hidden late-stage tubular damage).*