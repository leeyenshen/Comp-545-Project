Revised Experiment Design

Datasets (Updated): Replace the 2018-era QA sets with more recent, challenging benchmarks. For single-hop open QA, consider NaturalQuestions (NQ) (real Google queries, broad coverage, updated wiki content). For multi-hop reasoning, use MuSiQue (2022) – a 25k question dataset requiring 2–4 hops, which is significantly more difficult than HotpotQA (3× larger human–machine gap)
aclanthology.org
. These newer datasets ensure contemporary relevance and reduce the risk of evaluation on outdated facts. Additionally, explore specialized corpora for hallucination analysis: e.g. the RAGTruth corpus (ACL 2024), which contains ~18k LLM-generated RAG answers with manual hallucination annotations
aclanthology.org
. RAGTruth can serve as an evaluation benchmark to validate detectors under realistic settings (it reports that no current detector – not even GPT-4 – exceeds 50% precision and recall simultaneously
vectara.com
). This underscores the need for robust evaluation on up-to-date data.

Retrieval Tools (Non-LangChain): Implement retrieval using direct IR libraries for transparency and efficiency. For sparse retrieval, use Pyserini/Anserini (Lucene) to perform BM25 over the Wikipedia corpus (ensuring the wiki dump aligns with the dataset’s timeframe). For dense retrieval, interface with Faiss or ElasticSearch directly: e.g. encode passages with a strong bi-encoder (such as multi-embedding models or a DPR model) and index with Faiss – avoiding LangChain’s abstraction ensures we control retrieval quality and reproducibility. The retrieval pipeline will be tailored per dataset (e.g. NQ’s wiki passages or MuSiQue’s needed wiki index). We will explicitly vary retrieval quality by injecting distractors: for each query, mix a proportion of irrelevant passages into the retrieved set (e.g. 80% relevant vs 20% random for “high” quality, 50/50 for medium, and mostly irrelevant for “low” quality, following Niu et al.’s methodology). This controlled degradation simulates real-world retrieval failures. By not relying on LangChain, we can fine-tune BM25 parameters or embedding similarity thresholds as needed to achieve these target ratios accurately.

Hallucination Detectors (Benchmarked Methods): Expand and solidify the set of detection methods, grounding them in literature. We will implement three categories of detectors, ensuring each is properly validated:

Reference-based metrics: RAGAS – an open-source RAG evaluation suite that computes multi-faceted scores (e.g. context precision/recall, answer relevance, faithfulness)
cleanlab.ai
. RAGAS provides a holistic, reference-free hallucination score
cleanlab.ai
; we will use its Hallucination/Faithfulness sub-metrics to flag unsupported content. We’ll verify we have the latest version (as RAGAS is actively updated
arxiv.org
) and note any limitations (e.g. context utilization issues
cleanlab.ai
).

Entailment-based models: a NLI consistency model to judge if the generated answer is entailed by the retrieved context. We will use a state-of-the-art factuality checker (e.g. FactCC or a successor fine-tuned on summarization/QA consistency). Kryściński et al. (2020) introduced this approach for summary factuality; we’ll apply a similar model to RAG outputs. This provides a precision-oriented detector (likely to raise alerts when the answer conflicts with context, even if the answer is correct but unsupported).

Surface overlap metrics: Lexical overlap between answer and context (and/or embedding-based similarity). This simple baseline flags an answer as hallucinated if it contains many tokens or named entities not found in the retrieved text. While not robust to subtle errors, it offers a high-recall, low-precision check for obvious hallucinations. We will benchmark these methods on known data (e.g. verifying on RAGTruth’s annotated cases to see baseline performance) to ensure they behave as expected. Notably, proper benchmarking means using known annotated sets to calibrate detector thresholds and reporting metrics like AUROC, as done in recent work. We will also cite and discuss advanced detectors from recent literature (for Related Work): e.g. ReDeEP (Sun et al., 2024) which improves detection by separating the model’s use of external vs. internal knowledge
arxiv.org
, and LUMINA (Yeh et al., 2025) which tracks context–knowledge signals across transformer layers to achieve state-of-the-art detection AUROC (up to +13% vs prior methods)
arxiv.org
. These will guide our understanding, even if we don’t implement them fully, ensuring our report acknowledges cutting-edge approaches.

Timeline

Week 1: Data & Retrieval Setup (Dec 1–7): Finalize dataset choices and obtain data (e.g. download NQ and/or MuSiQue, subset if needed for feasibility). Build the Wikipedia index – use Pyserini to index the Wikipedia dump (for BM25) and set up Faiss with vector embeddings. Test retrieval quality on a small sample, and write a script to inject distractors at controlled ratios (validate that our “high/med/low quality” splits have the intended percentage of relevant vs. random passages). Milestone: by end of week, a working RAG pipeline that given a question can retrieve contexts at varying quality levels.

Week 2: LLM Response Generation (Dec 8–14): Integrate a suitable generator model. Use an open-source instruct-tuned LM like Llama-2-7B-chat or Mistral-7B (since Llama-3 was hypothetical). Ensure the model is set up in our environment (with GPU acceleration). Generate answers for a batch of questions under each retrieval quality condition. If time permits, generate multiple answers per condition to account for randomness. Start labeling results: compare model answers to gold answers to mark correctness/faithfulness (this provides ground truth for detector evaluation). Milestone: a dataset of model outputs labeled as factual/correct or hallucinated (unsupported/incorrect) for each retrieval setting.

Week 3: Detector Implementation & Initial Analysis (Dec 15–21): Implement and run the hallucination detection metrics on the generated outputs. Compute precision, recall, F1, and AUROC of each detector against our ground truth labels at each retrieval quality level. Perform error analysis: identify where detectors disagree or fail (e.g. a correct answer labeled hallucination by NLI due to missing support). Begin preparing visualizations (plots of detector performance vs. retrieval quality, confusion matrices). If possible, evaluate detectors on a small external benchmark (e.g. RAGTruth subset) for additional validation. Milestone: a complete set of results (tables/plots) and notes on key findings (which methods degrade most under low-quality retrieval, etc.).

Week 4: Writing & Polishing (Dec 22–28): Draft the final report in Overleaf using the CS224N format. Populate each section (Introduction, Methods, etc.) with content and insert figures. Refine the narrative to highlight our contributions: emphasize novel findings about detection robustness. Incorporate feedback from peers/instructors if available. Complete the Related Work with proper citations (ensuring we cite RAGTruth
aclanthology.org
, ReDeEP, LUMINA, etc. where relevant). Use the appendix for additional graphs or examples that didn’t fit in 6 pages. Finalize the bibliography. Milestone: Submit the polished report (~Dec 29), with a complete story from motivation to conclusion.

Proposed Report Structure

Abstract: (~200 words) Summarize the problem of hallucination detection in RAG and the gap addressed (robustness under poor retrieval). State our approach: “We systematically degrade retrieval quality and evaluate multiple detectors (LLM-based and heuristic) on QA tasks.” Highlight key result (detection performance drops sharply beyond a certain noise threshold) and significance (insights for deploying RAG in noisy settings).

Introduction: Introduce RAG and its promise to reduce hallucinations. Then note the overlooked issue: what if retrieval is imperfect? Ground this with context from prior work: retrieval quality strongly influences output correctness, yet few have studied the effect on hallucination detection. Clearly state our research question: “How does retrieval quality affect the reliability of hallucination detectors in RAG?”. Add a brief example or scenario illustrating a failure (e.g. a question where the retriever fails and the LLM makes up an answer that detectors miss). Finally, outline our contributions: e.g. “(1) We design a controlled experiment to vary retrieval noise, (2) evaluate three detection approaches (RAGAS, NLI-based, lexical) across conditions, (3) report how each degrades and analyze failure modes, and (4) provide recommendations for robust hallucination detection.”

Related Work: Discuss prior research on hallucination and detection in NLP. Start with general fact-checking and summarization factuality metrics (e.g. FactCC, QAGS, etc.), then focus on RAG-specific advancements. Cite RAGTruth (Niu et al., 2024) as a recent benchmark which found that even strong detectors (including GPT-4 as a judge) struggle to exceed 50% precision/recall
vectara.com
 – underscoring the problem’s difficulty. Mention approaches like ReDeEP (Sun et al., 2024) that use mechanistic interpretability to detect hallucinations by analyzing internal model behavior
arxiv.org
, and LUMINA (Yeh et al., 2025) that quantifies external vs. internal knowledge usage, achieving state-of-the-art detection performance
arxiv.org
. Also note tools like RAGAS (Es et al., 2025) for holistic RAG evaluation, and industry efforts (e.g. Vectara’s HHEM) highlighting the need for calibrated, efficient detectors. This section ensures the final report properly situates our work in context and fills the previous gap in citations.

Data & Environment: Describe the datasets and setup. For data: provide an overview of chosen QA dataset(s) – e.g. “NaturalQuestions open-domain: ~100k real queries, answers from Wikipedia; we use the official dev set for evaluation”. If multi-hop data (MuSiQue) is included, describe its scale and multi-hop nature. Clarify the retrieval corpus (e.g. Wikipedia dump from 2023) and how we index/search it (mention Pyserini/FAISS). List the LLM used for generation (e.g. “Llama-2-13B-chat, running on an A100 GPU” or similar) and any prompts or settings for generation. Also note how we simulate retrieval quality: e.g. “We create three versions of context for each query: high-quality (80% relevant docs), medium (50%), low (20% or none relevant), by mixing in random Wikipedia passages”. Include a brief table if space permits, illustrating stats like average retrieved relevant passages in each setting. Finally, mention any external resources or libraries (OpenAI API, HuggingFace models, etc.) used, and how we ensured reproducibility (fixed random seeds, etc.).

Methods: Detail our experimental pipeline. This can be structured in subsections:

RAG Pipeline: Explain how questions are processed: first the retrieval step (BM25 and dense retrieval strategies – give equations or descriptions for BM25 scoring, vector similarity, etc. as appropriate), then the generation step (the LLM takes question + retrieved passages as input and produces an answer). A diagram here is valuable (illustrating the flow: Question -> Retriever -> Retrieved docs -> LLM -> Answer). Emphasize the novel aspect: we inject noise into the retriever’s output to control quality. Possibly include a small schematic or algorithm for how distractors are sampled and combined.

Hallucination Detection Metrics: For each detector, describe how it works. For RAGAS: list the key metrics it uses (faithfulness: checks answer vs context support
cleanlab.ai
, answer relevancy, etc.) and note we aggregate these into an overall score or treat them separately. For the NLI-based detector: describe using a pre-trained entailment model to score the pair (context, answer) and classify as supported vs unsupported. For lexical overlap: define a simple metric (e.g. overlap% = |unique answer terms in context| / |unique terms in answer|, with a threshold). If we set any thresholds (like labeling hallucination if overlap% < X%), state those. This section should be methodological – how we apply each method to produce a label or score for a given QA pair. We should also mention the ground truth labeling procedure: since QA tasks have gold answers, we define an output as hallucinated if it is factually incorrect or unsupported by gold truth. (If the model’s answer exactly matches or paraphrases the gold answer, it’s faithful; otherwise it’s considered a hallucination for evaluation purposes. We’ll acknowledge complexities like an answer being correct but coming from parametric knowledge – which our detectors might flag – in the Discussion.)

Experimental Conditions: Reiterate that we evaluate each method under three retrieval conditions (high/med/low quality). This controlled setup is a key part of our method, ensuring we can attribute detection performance changes specifically to retrieval noise.

Experiments & Results: Present the outcomes of our study. Begin with a performance overview table: each row = a hallucination detector, columns = metrics (Precision, Recall, F1, AUROC) under High, Medium, Low retrieval quality. This will show the trend clearly (we expect numbers to worsen as quality drops). Highlight key findings in text: e.g. “Entailment-based detection saw precision plummet from 0.85 at high quality to 0.60 at low – indicating many false alarms when contexts were unreliable.” Meanwhile, perhaps lexical overlap has the opposite issue (missing many hallucinations until they are extreme). Include a figure (line chart) plotting AUROC vs. retrieval quality for each method, to visualize robustness: ideally, detectors with shallow slopes are robust, while steep drops indicate sensitivity to retrieval failures. We will also report any statistically significant differences if applicable (though with our sample size, this may be descriptive). Additionally, present confusion matrices or error breakdowns: for example, show how often each detector correctly vs. incorrectly flags hallucinations at low quality. If using an NLI model, we might show it often labels even correct answers as hallucinated when 0/5 retrieved docs contain the answer. If space allows, include a case study example: a small table showing a question, the low-quality retrieved passages (irrelevant), the LLM’s hallucinated answer, and how each detector scored it (illustrating, say, RAGAS gave a low faithfulness score correctly, but lexical overlap failed to flag it because some words overlapped by chance). Overall, this section should answer the core question: which detection methods hold up as retrieval degrades, and which fail, in what ways?

Discussion: Interpret the results and connect back to our hypotheses and broader context. Discuss why certain detectors behaved as they did. For instance, entailment detectors are overly pessimistic with noisy context, flagging even true answers as hallucinations (low precision) – confirming our hypothesis that they are most sensitive to misleading context. Simple overlap metrics might have high precision (only flagging when there’s zero overlap) but very low recall, missing subtle hallucinations – they might only catch extreme cases. RAGAS (multi-signal) might show more balanced performance, but we expect it too to degrade once irrelevant context confuses its LLM-based scoring. We should also discuss any unexpected findings: e.g. if medium quality retrieval sometimes fooled a detector more than low quality (perhaps partial context introduced confounders). Tie these observations to related work: e.g. RAGTruth’s finding of no perfect detector
vectara.com
 is mirrored in our results – all methods struggled to exceed 50% recall under worst-case retrieval. If we evaluated on RAGTruth data or others, mention how our results align or differ. Also, consider practical implications: for instance, in real systems, one might combine retrieval confidence with detection (our results suggest that if retrieval confidence is low, detectors might need adjusted thresholds or users should be warned regardless). We can propose ideas like training detectors on noisy-context data (as a future direction) to improve robustness. Finally, acknowledge limitations: due to time, we used smaller models (which might hallucinate more than GPT-4) and a limited set of detectors. We did not implement advanced methods like ReDeEP or LUMINA – our focus was evaluation, but those could further improve detection if incorporated. This frank discussion will strengthen the report’s credibility.

Conclusion: Recap the study and its main takeaways in a few sentences. For example: “We presented the first systematic evaluation of hallucination detection under varying retrieval quality in RAG. Our experiments show that as retrieval degrades, even state-of-the-art detectors (like LLM-based or entailment models) see significant drops in performance – in some cases failing to flag the majority of hallucinations. This stresses the importance of factoring in retrieval confidence when deploying RAG systems.” Note the implications (e.g. for building trustworthy systems, one must either improve retrieval or design detectors robust to its failures). End with potential future work: e.g. “exploring hybrid detectors that account for retrieval accuracy, or fine-tuning detection models on corpora like RAGTruth to improve resilience to noise.” Keep the tone optimistic that our findings will help guide such improvements.

(References will be included in a separate section, formatted in the conference style; key references are suggested below.)

Addressing Literature Gaps – Key Citations to Include

Niu et al. (2024) – RAGTruth: “RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models.” In ACL 2024, introduced a large manually-annotated dataset of RAG outputs (~18k) and benchmarked detection methods
aclanthology.org
. Findings: No detector (including GPT-4) achieved >50% precision and recall simultaneously on QA/summarization
vectara.com
. BibTex:```bib
@inproceedings{niu-etal-2024-ragtruth,
title = "{RAGT}ruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models",
author = "Niu, Cheng and Wu, Yuanhao and Zhu, Juno and Xu, Siliang and Shum, KaShun and Zhong, Randy and Song, Juntong and Zhang, Tong",
booktitle = "Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers)",
year = "2024",
pages = "10669--10685",
url = "https://aclanthology.org/2024.acl-long.585"
}


- **Sun et al. (2024) – ReDeEP:** *“ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability.”* (arXiv preprint 2410.11414, under review for ICLR 2026). Proposes a novel detector analyzing *which* transformer components cause hallucination (identifies over-reliance on parametric knowledge vs. failure to use retrieved content):contentReference[oaicite:28]{index=28}. **BibTex:**```bib
@article{sun2024redeep,
  title={ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability},
  author={Sun, Zhongxiang and Zang, Xiaoxue and Zheng, Kai and Song, Yang and Xu, Jun and Zhang, Xiao and Yu, Weijie and Li, Han},
  journal={arXiv preprint arXiv:2410.11414},
  year={2024}
}


Yeh et al. (2025) – LUMINA: “LUMINA: Detecting hallucinations in RAG systems with context–knowledge signals.” (arXiv 2509.21875). Develops a detection framework that quantifies external context utilization (distributional difference if context is ignored) and internal knowledge use (tracking token evolution in the model)
arxiv.org
. Achieves top performance on multiple benchmarks, improving AUROC by ~13% over prior methods
arxiv.org
. BibTex:```bib
@article{yeh2025lumina,
title={LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals},
author={Yeh, Min-Hsuan and Li, Yixuan and Mallick, Tanwi},
journal={arXiv preprint arXiv:2509.21875},
year={2025}
}


- **Es et al. (2025) – RAGAS:** *“RAGAS: Automated Evaluation of Retrieval-Augmented Generation.”* (arXiv 2309.15217, v2 2025). Proposes an evaluation suite for RAG pipelines with metrics for context relevance, faithfulness, answer correctness, etc., enabling reference-free hallucination detection:contentReference[oaicite:31]{index=31}. We use RAGAS in our method; citing this ensures proper attribution. **BibTex:**```bib
@article{es2025ragas,
  title={RAGAS: Automated Evaluation of Retrieval Augmented Generation},
  author={Es, Shahul and James, Jithin and Espinosa-Anke, Luis and Schockaert, Steven},
  journal={arXiv preprint arXiv:2309.15217},
  year={2025}
}


(Including these references in the final report will address the instructor’s feedback on missing citations, and strengthen the related work section.)

Required Figures and Visuals

Pipeline Diagram: A schematic illustrating our RAG setup. Show the flow from Question -> Retriever -> Good/Poor retrieved context -> LLM -> Answer, and then -> Detector(s). This helps readers visualize the experiment design. It can be annotated to show the “knob” we turn (retrieval quality percentage). For example, use color coding or dashed arrows to indicate high vs. low quality retrieval paths.

Performance Plot: A line chart (or bar chart) showing each detector’s performance vs. retrieval quality. For instance, x-axis = % of relevant documents in context (80%, 50%, 20%), y-axis = AUROC (or F1). Plot separate lines for RAGAS, NLI model, and Lexical overlap. We expect downward slopes; the figure will make the trend immediately clear and allow quick comparison of which method degrades fastest.

Table of Results: A concise table (could be in-text or in appendix) listing numerical results (Precision/Recall/F1/AUROC) for each detector under each condition. This provides exact values complementing the performance plot. Highlight significant drops (e.g. by bolding worst-case values) to draw attention to severe degradation.

Example Case Study: A small figure or diagram illustrating a concrete example of hallucination detection (likely for the Discussion or Appendix). For instance, a side-by-side comparison: Case: Question: “Who won the 2025 Nobel Prize in Physics?”; High-quality retrieval: (relevant wiki passage) → model answer (correct) → detectors all clear; Low-quality retrieval: (irrelevant text) → model answer (hallucinated name) → detectors’ outputs (maybe NLI flags it, lexical does not, etc.). This could be presented as a flow or as a highlighted text snippet with annotations (to show which parts of the answer were unsupported). Real examples will make the failure modes more tangible.

Confusion Matrix or Bar chart of Errors: If space permits, a visual breakdown of error types. For each detector, illustrate False Positives and False Negatives at low retrieval quality. E.g., a pair of bar charts showing the count of each outcome (TP, FP, FN, TN) for high vs. low quality. This emphasizes how, say, false negatives spike for the lexical method (missing many hallucinations) or false positives spike for the entailment method (flagging non-hallucinations when context is bad). Such visualization can enrich the Discussion section.

Each figure will be referenced in the text at the appropriate point (Experiments or Discussion) to support our analysis. We will ensure they are high-clarity and conform to the 6-page format (using color or patterns distinguishably, and placing any large charts in the appendix if needed). These visuals collectively will greatly enhance the reader’s understanding of our approach and findings, making the final report more compelling and clear.