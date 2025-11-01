package com.chethan.replicationcontroller.dto;

import lombok.Data;

import java.util.Map;

@Data
public class NodeMetric {
    private String nodeId;
    private Map<String, KeyMetric> keyMetrics;
    private double storageCost;
}
