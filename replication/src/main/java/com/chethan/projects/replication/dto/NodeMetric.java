package com.chethan.projects.replication.dto;

import lombok.Data;
import java.util.Map;

@Data
public class NodeMetric {
    private String nodeId;
    // The key of the map is the data key (e.g., "key1")
    private Map<String, KeyMetric> keyMetrics;
    private double storageCost;
}
