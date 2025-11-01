package com.chethan.projects.replication.config;

public final class CostConstants {

    // Latency in milliseconds
    public static final long LOCAL_READ_LATENCY_MS = 10;
    public static final long REMOTE_READ_LATENCY_MS = 150;

    // Cost in a hypothetical currency unit (e.g., dollars)
    public static final double COST_PER_KEY_STORED = 1.5;

    private CostConstants() {
    }
}