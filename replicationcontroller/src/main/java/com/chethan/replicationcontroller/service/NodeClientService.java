package com.chethan.replicationcontroller.service;

import com.chethan.replicationcontroller.dto.NodeMetric;
import com.chethan.replicationcontroller.dto.NodeReadResponse;
import com.chethan.replicationcontroller.dto.ReplicationRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class NodeClientService {

    private static final Logger logger = LoggerFactory.getLogger(NodeClientService.class);

    @Autowired
    private RestTemplate restTemplate;

    public void replicateData(String nodeUrl, String key, String value) {
        String url = nodeUrl + "/management/replicate";
        try {
            // The DB Node's simple replicate endpoint expects this body
            ReplicationRequest request = new ReplicationRequest(key, value);
            restTemplate.postForEntity(url, request, Void.class);
            logger.info("Successfully replicated key '{}' to node {}", key, nodeUrl);
        } catch (Exception e) {
            // In a real system, we'd have retry logic or a queue
            logger.error("Failed to replicate key '{}' to node {}: {}", key, nodeUrl, e.getMessage());
        }
    }

    public NodeReadResponse fetchData(String nodeUrl, String key) {
        String url = nodeUrl + "/data/" + key;
        try {
            ResponseEntity<NodeReadResponse> response = restTemplate.getForEntity(url, NodeReadResponse.class);
            return response.getBody();
        } catch (Exception e) {
            logger.error("Failed to fetch key '{}' from node {}: {}", key, nodeUrl, e.getMessage());
            // Return a null or a special error object to indicate failure
            return null;
        }
    }

    public NodeMetric getMetrics(String nodeUrl) {
        String url = nodeUrl + "/management/metrics";
        try {
            return restTemplate.getForObject(url, NodeMetric.class);
        } catch (Exception e) {
            logger.error("Failed to get metrics from node {}: {}", nodeUrl, e.getMessage());
            return null; // Handle failure
        }
    }

    public void evictData(String nodeUrl, String key) {
        String url = nodeUrl + "/management/data/" + key;
        try {
            restTemplate.delete(url);
            logger.info("Successfully evicted key '{}' from node {}", key, nodeUrl);
        } catch (Exception e) {
            logger.error("Failed to evict key '{}' from node {}: {}", key, nodeUrl, e.getMessage());
        }
    }
}
