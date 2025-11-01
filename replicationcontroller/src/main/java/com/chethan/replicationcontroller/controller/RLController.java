package com.chethan.replicationcontroller.controller;

import com.chethan.replicationcontroller.config.ClusterConfig;
import com.chethan.replicationcontroller.dto.NodeMetric;
import com.chethan.replicationcontroller.dto.RLActionRequest;
import com.chethan.replicationcontroller.service.NodeClientService;
import com.chethan.replicationcontroller.service.ReplicationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/rl")
public class RLController {

    @Autowired
    private NodeClientService nodeClientService;
    @Autowired
    private ClusterConfig clusterConfig;

    @Autowired
    private ReplicationService replicationService;

    // A simple map to convert service names (e.g., "replication-us") to URLs
    private final Map<String, String> nodeNameToUrlMap = Map.of(
            "replication-us", "http://replication-us:8080",
            "replication-eu", "http://replication-eu:8080",
            "replication-ap", "http://replication-ap:8080"
    );

    @PostMapping("/execute-action")
    public ResponseEntity<Void> executeAction(@RequestBody RLActionRequest actionRequest) {
        String targetNodeUrl = nodeNameToUrlMap.get(actionRequest.getTargetNode());
        if (targetNodeUrl != null) {
            replicationService.executeAction(
                    actionRequest.getActionType(),
                    actionRequest.getKey(),
                    targetNodeUrl
            );
            return ResponseEntity.accepted().build();
        }
        return ResponseEntity.badRequest().build(); // Node name not found
    }

    @GetMapping("/system-state")
    public ResponseEntity<List<NodeMetric>> getSystemState() {
        List<NodeMetric> allMetrics = clusterConfig.getNodes().stream()
                .map(nodeUrl -> nodeClientService.getMetrics(nodeUrl))
                .filter(Objects::nonNull) // Filter out any nodes that failed to respond
                .collect(Collectors.toList());
        return ResponseEntity.ok(allMetrics);
    }
}