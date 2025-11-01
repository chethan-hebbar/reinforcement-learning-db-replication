package com.chethan.projects.replication;

import com.chethan.projects.replication.dto.ReadResponse;
import com.chethan.projects.replication.service.DataStoreService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DataStoreServiceTest {

    private DataStoreService dataStoreService;

    @BeforeEach
    void setUp() {
        // Create a new instance before each test to ensure isolation
        dataStoreService = new DataStoreService();
    }

    @Test
    void testPutAndGet() {
        // Action
        dataStoreService.put("key1", "value1");
        ReadResponse response = dataStoreService.handleGet("key1");

        // Assertion
        assertEquals("value1", response.getValue());
        assertTrue(response.getLatencyMs() > 0);
        assertEquals(1, dataStoreService.getReadCount("key1"));
        assertEquals(1, dataStoreService.getWriteCount("key1"));
    }

    @Test
    void testEvict() {
        // Setup
        dataStoreService.put("key1", "value1");
        assertTrue(dataStoreService.contains("key1"));

        // Action
        dataStoreService.evict("key1");

        // Assertion
        assertFalse(dataStoreService.contains("key1"));
    }

    @Test
    void testReadMiss() {
        // Action: Attempt to read a key that doesn't exist
        ReadResponse response = dataStoreService.handleGet("nonexistent_key");

        // Assertion
        assertNull(response.getValue());
        assertTrue(response.getLatencyMs() > 100); // Should be remote latency
        assertEquals(1, dataStoreService.getReadCount("nonexistent_key"));
        assertEquals(0, dataStoreService.getWriteCount("nonexistent_key"));
    }

    @Test
    void testStorageCost() {
        // Action
        dataStoreService.put("keyA", "valA");
        dataStoreService.put("keyB", "valB");

        // Assertion
        // Assumes COST_PER_KEY_STORED is 1.5
        assertEquals(3.0, dataStoreService.getStorageCost());
    }
}