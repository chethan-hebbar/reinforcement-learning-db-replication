package com.chethan.replicationcontroller.dto;
import lombok.Data;

@Data
public class RLActionRequest {
    private String actionType; // "REPLICATE" or "EVICT"
    private String key;
    private String targetNode; // The node to act upon
}