# CS 6804  Project: Graph Neural Network-based Anomaly Detection for Sensor Networks

Recent advances in the field of the Internet of Things (IoT) have fostered the capability to collect and analyze large amounts of data across IoT domains such as advanced manufacturing, smart city,  etc.
However, there is an increasing need to monitor these devices to secure the system against attacks or identify the failures of sensors during the process. 
However, due to the unique data structure and complex correlation between variables in a dynamic network, it is challenging to detect anomalous events with high-dimensional time series data from multiple sensors.
In this project, I investigate graph neural network-based (GNN) anomaly detection in sensor networks. The prior information is utilized to learn the correlation and facilitate the monitoring of multiple sensors.
Also, comparisons are conducted to investigate the advantage of GNN over traditional neural networks-based approaches such as Long Short Term Memory (LSTM)-Variational Autoencode (VAE).

* **Dataset**: [Intel Berkeley Research Labs (IBRL)](http://db.csail.mit.edu/labdata/labdata.html)
* **Approach**:
  * GNN folder: [Graph Deviation Network](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/download/16523/16330&hl=en&sa=T&oi=gsb-gga&ct=res&cd=0&d=12130066713245760732&ei=GrNaZNXvFanZsQLh8KKoAQ&scisig=AGlGAw9VoG9Rx9HkKA1_EDDAoNoQ)
  * pyodds folder: [LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection](https://arxiv.org/pdf/1607.00148.pdf)

    