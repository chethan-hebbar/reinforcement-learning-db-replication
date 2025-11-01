package com.chethan.projects.replication.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ReadResponse {
    private String key;
    private String value;
    private long latencyMs;
}