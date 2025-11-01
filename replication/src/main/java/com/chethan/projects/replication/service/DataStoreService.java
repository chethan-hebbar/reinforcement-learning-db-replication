package com.chethan.projects.replication.service;

import com.chethan.projects.replication.config.CostConstants;
import com.chethan.projects.replication.dto.KeyMetric;
import com.chethan.projects.replication.dto.ReadResponse;
import org.springframework.stereotype.Service;

import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;

@Service
public class DataStoreService {

    private final ConcurrentHashMap<String, String> store = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, LongAdder> readCounts = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, LongAdder> writeCounts = new ConcurrentHashMap<>();


    public void put(String key, String value) {
        store.put(key, value);
        // Increment write count for this key
        writeCounts.computeIfAbsent(key, k -> new LongAdder()).increment();
    }

    public Optional<String> get(String key) {
        if (store.containsKey(key)) {
            // Increment read count only on a successful read (a hit)
            readCounts.computeIfAbsent(key, k -> new LongAdder()).increment();
            return Optional.ofNullable(store.get(key));
        }
        return Optional.empty(); // Data not found
    }

    public void evict(String key) {
        store.remove(key);
        readCounts.remove(key);
        writeCounts.remove(key);
    }

    public boolean contains(String key) {
        return store.containsKey(key);
    }

    // We will need a way to get these metrics later for the API
    public long getReadCount(String key) {
        return Optional.ofNullable(readCounts.get(key)).map(LongAdder::sum).orElse(0L);
    }

    public long getWriteCount(String key) {
        return Optional.ofNullable(writeCounts.get(key)).map(LongAdder::sum).orElse(0L);
    }

    public Map<String, KeyMetric> getAllKeyMetrics() {
        // Collect all unique keys from all maps
        Set<String> allKeys = new HashSet<>(store.keySet());
        allKeys.addAll(readCounts.keySet());
        allKeys.addAll(writeCounts.keySet());

        return allKeys.stream()
                .collect(Collectors.toMap(
                        key -> key,
                        key -> new KeyMetric(getReadCount(key), getWriteCount(key))
                ));
    }

    /**
     * Performs a read operation, simulating latency and tracking metrics.
     * This method will be called by our public-facing API.
     * @param key The key to read.
     * @return A ReadResponse containing the value and simulated latency.
     */
    public ReadResponse handleGet(String key) {
        try {
            if (store.containsKey(key)) {
                // --- LOCAL HIT ---
                Thread.sleep(CostConstants.LOCAL_READ_LATENCY_MS); // Simulate latency
                readCounts.computeIfAbsent(key, k -> new LongAdder()).increment();
                return new ReadResponse(key, store.get(key), CostConstants.LOCAL_READ_LATENCY_MS);
            } else {
                // --- MISS (requires a remote fetch) ---
                Thread.sleep(CostConstants.REMOTE_READ_LATENCY_MS); // Simulate high latency
                // We still increment the read count for the key, as a read was attempted.
                readCounts.computeIfAbsent(key, k -> new LongAdder()).increment();
                return new ReadResponse(key, null, CostConstants.REMOTE_READ_LATENCY_MS);
            }
        }
        catch (Exception e) {
            Thread.currentThread().interrupt();
            // Return an error or an empty response
            return new ReadResponse(key, null, 0);
        }
    }

    /**
     * Calculates the total storage cost for the node.
     * @return The calculated storage cost.
     */
    public double getStorageCost() {
        return store.size() * CostConstants.COST_PER_KEY_STORED;
    }
}