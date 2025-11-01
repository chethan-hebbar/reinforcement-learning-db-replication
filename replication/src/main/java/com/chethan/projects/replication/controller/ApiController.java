package com.chethan.projects.replication.controller;

import com.chethan.projects.replication.dto.KeyMetric;
import com.chethan.projects.replication.dto.NodeMetric;
import com.chethan.projects.replication.dto.ReadResponse;
import com.chethan.projects.replication.dto.SimpleReplicationRequest;
import com.chethan.projects.replication.service.DataStoreService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Optional;

@RestController
public class ApiController {

    @Autowired
    private DataStoreService dataStoreService;

    // Inject the node ID from application.properties
    @Value("${node.id}")
    private String nodeId;

    /**
     * INTERNAL API: Used by another DB node to fetch data for replication.
     */
    @GetMapping("/internal/data/{key}")
    public ResponseEntity<String> getInternalData(@PathVariable String key) {
        Optional<String> value = dataStoreService.get(key);
        // Note: This does NOT increment the public read count, as it's an internal operation.
        // If we wanted to, we would call a different service method.
        return value.map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /**
     * MANAGEMENT API: Used by the Controller to evict a key.
     */
    @DeleteMapping("/management/data/{key}")
    public ResponseEntity<Void> evictData(@PathVariable String key) {
        dataStoreService.evict(key);
        return ResponseEntity.ok().build();
    }

    /**
     * MANAGEMENT API: Used by the Controller to get all node metrics.
     */
    @GetMapping("/management/metrics")
    public ResponseEntity<NodeMetric> getMetrics() {
        NodeMetric metrics = new NodeMetric();
        metrics.setNodeId(nodeId);
        Map<String, KeyMetric> keyMetrics = dataStoreService.getAllKeyMetrics();
        metrics.setKeyMetrics(keyMetrics);

        metrics.setStorageCost(dataStoreService.getStorageCost());

        return ResponseEntity.ok(metrics);
    }

    /**
     * MANAGEMENT API: A simple way to add data for testing.
     * Later, this will be expanded to fetch from a source node.
     */
    @PostMapping("/management/replicate")
    public ResponseEntity<Void> replicateData(@RequestBody SimpleReplicationRequest request) {
        // This is a simplified version for now.
        dataStoreService.put(request.getKey(), request.getValue());
        return ResponseEntity.ok().build();
    }

    @GetMapping("/data/{key}")
    public ResponseEntity<ReadResponse> getData(@PathVariable String key) {
        ReadResponse response = dataStoreService.handleGet(key);
        if (response.getValue() == null) {
            // We return 200 OK even on a miss, because the response body
            // contains the crucial latency information.
            return ResponseEntity.ok(response);
        }
        return ResponseEntity.ok(response);
    }
}
