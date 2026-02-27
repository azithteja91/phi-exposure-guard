

```mermaid
flowchart LR

%% Input Layer
A["Streaming Inputs<br/>Text · ASR · Image · Waveform · Audio"] --> B["Modality Parsers"]

%% Exposure Extraction
B --> C["Identifier Detection<br/>per modality"]

%% Cross-Modal Aggregation
C --> D["Entity Resolver<br/>Cross-Modal Linking"]

%% Longitudinal State
D --> E["Exposure State Store<br/>Persistent Entity Memory"]

%% Risk Modeling
E --> F["Risk Estimator<br/>Cumulative Exposure Score"]

%% Adaptive Control
F --> G["Adaptive Policy Controller"]

%% Masking Options
G -->|Low Exposure| H["Weak / Light Masking"]
G -->|Moderate Exposure| I["Consistent Pseudonymization"]
G -->|High Exposure| J["Strong Redaction / Removal"]

%% Output Stream
H --> K["Sanitized Output Stream"]
I --> K
J --> K

%% Closed Loop
K --> L["Post-Masking Update"]
L --> E

%% Audit & Reproducibility
G --> M["Decision Log"]
M --> N["Audit Records<br/>Replayable Execution"]
