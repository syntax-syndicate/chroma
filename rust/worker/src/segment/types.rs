use crate::types::EmbeddingRecord;

trait SegmentImpl {
    fn begin_transaction(&self);
    fn write_records(&self, records: Vec<Box<EmbeddingRecord>>, offset_ids: Vec<u32>);
    fn commit_transaction(&self);
    fn rollback_transaction(&self);
}

trait OffsetIdAssigner {
    fn assign_offset_ids(&self, records: Vec<Box<EmbeddingRecord>>) -> Vec<u32>;
}
