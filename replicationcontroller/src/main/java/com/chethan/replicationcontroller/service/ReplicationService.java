package com.chethan.replicationcontroller.service;

import com.chethan.replicationcontroller.config.ClusterConfig;
import com.chethan.replicationcontroller.dto.ClientReadResponse;
import com.chethan.replicationcontroller.dto.NodeReadResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
@Slf4j
@Service
public class ReplicationService {

    // Maps a data key to a set of node URLs that hold the data.
    // e.g., "user123" -> {"http://localhost:8081", "http://localhost:8082"}
    private final ConcurrentHashMap<String, Set<String>> replicationMap = new ConcurrentHashMap<>();

    @Autowired
    private ClusterConfig clusterConfig;

    @Autowired
    private NodeClientService nodeClientService;

    /**
     * Handles a write request based on the static replication policy.
     * Policy: Replicate to all nodes.
     */
    public void handleWrite(String key, String value) {
        // For each node in the cluster, send a replicate command
        for (String nodeUrl : clusterConfig.getNodes()) {
            nodeClientService.replicateData(nodeUrl, key, value);
        }

        // After successfully commanding all nodes, update our internal map
        replicationMap.put(key, new HashSet<>(clusterConfig.getNodes()));
    }

    /**
     * Returns the set of nodes where a key is replicated.
     */
    public Set<String> getNodesForKey(String key) {
        return replicationMap.getOrDefault(key, Collections.emptySet());
    }

    /**
     * Handles a read request by routing it to an appropriate node.
     * @param key The key to read.
     * @return The response to be sent to the client.
     */
    public ClientReadResponse handleRead(String key) {
        Set<String> nodesWithKey = getNodesForKey(key);

        String targetNode;
        if (nodesWithKey.isEmpty()) {
            // --- CACHE MISS SCENARIO ---
            // The controller doesn't know where the data is. This can happen if the
            // controller restarts or the data was written before our current policy.
            // We'll just pick the first node in the cluster to ask. This will result
            // in a high-latency remote read on the DB-node side.
            log.warn("Key '{}' not found in replication map. Performing a discovery read.", key);
            targetNode = clusterConfig.getNodes().getFirst();
        } else {
            // --- CACHE HIT SCENARIO ---
            // For now, we'll use a simple strategy: just pick the first node from the set.
            // A more advanced strategy could be to pick one randomly for load balancing,
            // or pick the one geographically closest to the user.
            targetNode = nodesWithKey.iterator().next();
        }

        log.info("Routing read for key '{}' to node {}", key, targetNode);
        NodeReadResponse nodeResponse = nodeClientService.fetchData(targetNode, key);

        if (nodeResponse == null) {
            // Handle case where the node is down or request fails
            return new ClientReadResponse(key, null, 0, "ERROR: FAILED_TO_FETCH");
        }

        // The value might be null if it was a remote read on the db-node, which is expected.
        return new ClientReadResponse(
                key,
                nodeResponse.getValue(),
                nodeResponse.getLatencyMs(),
                targetNode
        );
    }

    public void executeAction(String actionType, String key, String targetNodeUrl) {
        if ("REPLICATE".equalsIgnoreCase(actionType)) {
            // For replication, the source of truth is the first node that has the key.
            // A more robust implementation would be needed for production.
            // In our simulation, all nodes get all writes, so any node is a valid source.
            String sourceNodeUrl = clusterConfig.getNodes().getFirst(); // Simplification for now

            // We need to update the NodeClientService to handle this more complex replication
            // For now, let's assume a simplified 'put' command for the agent's action
            nodeClientService.replicateData(targetNodeUrl, key, "agent-replicated-value"); // Value is a placeholder

            // Update the replication map
            getNodesForKey(key).add(targetNodeUrl);

        }
        else if ("EVICT".equalsIgnoreCase(actionType)) {
            nodeClientService.evictData(targetNodeUrl, key);
            // Update the replication map
            getNodesForKey(key).remove(targetNodeUrl);
        }
    }
}