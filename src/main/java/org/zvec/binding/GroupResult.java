package org.zvec.binding;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One group of a grouped vector search, available since zvec v0.8.1.
 *
 * <p>Documents are ordered by vector similarity inside the group and are owned
 * by this object, so they stay readable after the query and the collection are
 * closed. Release them with {@link #close()} or
 * {@link #freeGroupResults(List)}.
 *
 * @see Collection#groupByQuery(GroupByVectorQuery)
 */
public class GroupResult implements AutoCloseable {

    private final String groupByValue;
    private final List<Doc> docs;

    GroupResult(String groupByValue, List<Doc> docs) {
        this.groupByValue = groupByValue;
        this.docs = Collections.unmodifiableList(new ArrayList<>(docs));
    }

    /** String representation of the value the group was formed on. */
    public String getGroupByValue() {
        return groupByValue;
    }

    /** Documents in the group, closest match first. Owned by this result. */
    public List<Doc> getDocs() {
        return docs;
    }

    /** Number of documents in this group. */
    public int getDocCount() {
        return docs.size();
    }

    /** Destroy every document in this group. Safe to call more than once. */
    public void destroy() {
        Doc.freeDocs(docs);
    }

    @Override
    public void close() {
        destroy();
    }

    /** Destroy the documents of several grouped results at once. */
    public static void freeGroupResults(List<GroupResult> groups) {
        if (groups == null) return;
        for (GroupResult group : groups) {
            if (group != null) group.destroy();
        }
    }

    @Override
    public String toString() {
        return "GroupResult{groupByValue='" + groupByValue + "', docCount=" + docs.size() + '}';
    }
}
