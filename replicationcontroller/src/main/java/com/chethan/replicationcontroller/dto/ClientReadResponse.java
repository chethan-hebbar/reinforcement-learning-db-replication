package com.chethan.replicationcontroller.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ClientReadResponse {
    private String key;
    private String value;
    private long retrievalLatencyMs;
    private String servedByNode;
}