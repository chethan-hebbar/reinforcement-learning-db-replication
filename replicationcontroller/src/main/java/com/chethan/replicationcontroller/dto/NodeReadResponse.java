package com.chethan.replicationcontroller.dto;

import lombok.Data;

@Data
public class NodeReadResponse {
    private String key;
    private String value;
    private long latencyMs;
}
