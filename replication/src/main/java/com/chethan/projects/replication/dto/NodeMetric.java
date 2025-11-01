package com.chethan.projects.replication.dto;

import lombok.Data;
import java.util.Map;

@Data
public class NodeMetric {
    private String nodeId;
    private Map<String, KeyMetric> keyMetrics;
    private double storageCost;
}
