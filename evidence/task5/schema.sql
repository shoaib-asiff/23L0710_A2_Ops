
CREATE TABLE prediction_records (
	id INTEGER NOT NULL, 
	timestamp DATETIME NOT NULL, 
	filename VARCHAR(255) NOT NULL, 
	prediction VARCHAR(50) NOT NULL, 
	confidence FLOAT NOT NULL, 
	raw_score FLOAT NOT NULL, 
	threshold FLOAT NOT NULL, 
	model_name VARCHAR(100) NOT NULL, 
	inference_mode VARCHAR(20) NOT NULL, 
	inference_time_ms FLOAT NOT NULL, 
	client_ip VARCHAR(50), 
	PRIMARY KEY (id)
)

