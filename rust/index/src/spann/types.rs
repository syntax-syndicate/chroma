use std::{
    collections::HashMap,
    sync::{atomic::AtomicU32, Arc},
};

use chroma_blockstore::{
    provider::BlockfileProvider, BlockfileFlusher, BlockfileWriter, BlockfileWriterOptions,
};
use chroma_distance::DistanceFunction;
use chroma_error::{ChromaError, ErrorCodes};
use chroma_types::CollectionUuid;
use chroma_types::SpannPostingList;
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use thiserror::Error;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    hnsw_provider::{HnswIndexProvider, HnswIndexRef},
    utils::normalize,
    Index, IndexUuid,
};

use super::utils::KMeansAlgorithm;

pub struct VersionsMapInner {
    pub versions_map: HashMap<u32, u32>,
}

#[allow(dead_code)]
// Note: Fields of this struct are public for testing.
pub struct SpannIndexWriter {
    // HNSW index and its provider for centroid search.
    pub hnsw_index: HnswIndexRef,
    hnsw_provider: HnswIndexProvider,
    blockfile_provider: BlockfileProvider,
    // Posting list of the centroids.
    // TODO(Sanket): For now the lock is very coarse grained. But this should
    // be changed in future if perf is not satisfactory.
    pub posting_list_writer: Arc<Mutex<BlockfileWriter>>,
    pub next_head_id: Arc<AtomicU32>,
    // Version number of each point.
    // TODO(Sanket): Finer grained locking for this map in future if perf is not satisfactory.
    pub versions_map: Arc<RwLock<VersionsMapInner>>,
    pub distance_function: DistanceFunction,
    pub dimensionality: usize,
}

#[derive(Error, Debug)]
pub enum SpannIndexWriterConstructionError {
    #[error("Error creating/forking hnsw index")]
    HnswIndexConstructionError,
    #[error("Error creating blockfile reader")]
    BlockfileReaderConstructionError,
    #[error("Error creating/forking blockfile writer")]
    BlockfileWriterConstructionError,
    #[error("Error loading version data from blockfile")]
    BlockfileVersionDataLoadError,
    #[error("Error resizing hnsw index")]
    HnswIndexResizeError,
    #[error("Error adding to hnsw index")]
    HnswIndexAddError,
    #[error("Error searching from hnsw")]
    HnswIndexSearchError,
    #[error("Error adding to posting list")]
    PostingListAddError,
    #[error("Error searching for posting list")]
    PostingListSearchError,
    #[error("Expected data not found")]
    ExpectedDataNotFound,
}

impl ChromaError for SpannIndexWriterConstructionError {
    fn code(&self) -> ErrorCodes {
        match self {
            Self::HnswIndexConstructionError => ErrorCodes::Internal,
            Self::BlockfileReaderConstructionError => ErrorCodes::Internal,
            Self::BlockfileWriterConstructionError => ErrorCodes::Internal,
            Self::BlockfileVersionDataLoadError => ErrorCodes::Internal,
            Self::HnswIndexResizeError => ErrorCodes::Internal,
            Self::HnswIndexAddError => ErrorCodes::Internal,
            Self::PostingListAddError => ErrorCodes::Internal,
            Self::HnswIndexSearchError => ErrorCodes::Internal,
            Self::PostingListSearchError => ErrorCodes::Internal,
            Self::ExpectedDataNotFound => ErrorCodes::Internal,
        }
    }
}

const MAX_HEAD_OFFSET_ID: &str = "max_head_offset_id";

// TODO(Sanket): Make these configurable.
#[allow(dead_code)]
const NUM_CENTROIDS_TO_SEARCH: u32 = 64;
#[allow(dead_code)]
const RNG_FACTOR: f32 = 1.0;
#[allow(dead_code)]
const SPLIT_THRESHOLD: usize = 100;
const NUM_SAMPLES_FOR_KMEANS: usize = 1000;
const INITIAL_LAMBDA: f32 = 100.0;

impl SpannIndexWriter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hnsw_index: HnswIndexRef,
        hnsw_provider: HnswIndexProvider,
        blockfile_provider: BlockfileProvider,
        posting_list_writer: BlockfileWriter,
        next_head_id: u32,
        versions_map: VersionsMapInner,
        distance_function: DistanceFunction,
        dimensionality: usize,
    ) -> Self {
        SpannIndexWriter {
            hnsw_index,
            hnsw_provider,
            blockfile_provider,
            posting_list_writer: Arc::new(Mutex::new(posting_list_writer)),
            next_head_id: Arc::new(AtomicU32::new(next_head_id)),
            versions_map: Arc::new(RwLock::new(versions_map)),
            distance_function,
            dimensionality,
        }
    }

    async fn hnsw_index_from_id(
        hnsw_provider: &HnswIndexProvider,
        id: &IndexUuid,
        collection_id: &CollectionUuid,
        distance_function: DistanceFunction,
        dimensionality: usize,
    ) -> Result<HnswIndexRef, SpannIndexWriterConstructionError> {
        match hnsw_provider
            .fork(id, collection_id, dimensionality as i32, distance_function)
            .await
        {
            Ok(index) => Ok(index),
            Err(_) => Err(SpannIndexWriterConstructionError::HnswIndexConstructionError),
        }
    }

    async fn create_hnsw_index(
        hnsw_provider: &HnswIndexProvider,
        collection_id: &CollectionUuid,
        distance_function: DistanceFunction,
        dimensionality: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Result<HnswIndexRef, SpannIndexWriterConstructionError> {
        match hnsw_provider
            .create(
                collection_id,
                m,
                ef_construction,
                ef_search,
                dimensionality as i32,
                distance_function,
            )
            .await
        {
            Ok(index) => Ok(index),
            Err(_) => Err(SpannIndexWriterConstructionError::HnswIndexConstructionError),
        }
    }

    async fn load_versions_map(
        blockfile_id: &Uuid,
        blockfile_provider: &BlockfileProvider,
    ) -> Result<VersionsMapInner, SpannIndexWriterConstructionError> {
        // Create a reader for the blockfile. Load all the data into the versions map.
        let mut versions_map = HashMap::new();
        let reader = match blockfile_provider.read::<u32, u32>(blockfile_id).await {
            Ok(reader) => reader,
            Err(_) => {
                return Err(SpannIndexWriterConstructionError::BlockfileReaderConstructionError)
            }
        };
        // Load data using the reader.
        let versions_data = reader
            .get_range(.., ..)
            .await
            .map_err(|_| SpannIndexWriterConstructionError::BlockfileVersionDataLoadError)?;
        versions_data.iter().for_each(|(key, value)| {
            versions_map.insert(*key, *value);
        });
        Ok(VersionsMapInner { versions_map })
    }

    async fn fork_postings_list(
        blockfile_id: &Uuid,
        blockfile_provider: &BlockfileProvider,
    ) -> Result<BlockfileWriter, SpannIndexWriterConstructionError> {
        let mut bf_options = BlockfileWriterOptions::new();
        bf_options = bf_options.unordered_mutations();
        bf_options = bf_options.fork(*blockfile_id);
        match blockfile_provider
            .write::<u32, &SpannPostingList<'_>>(bf_options)
            .await
        {
            Ok(writer) => Ok(writer),
            Err(_) => Err(SpannIndexWriterConstructionError::BlockfileWriterConstructionError),
        }
    }

    async fn create_posting_list(
        blockfile_provider: &BlockfileProvider,
    ) -> Result<BlockfileWriter, SpannIndexWriterConstructionError> {
        let mut bf_options = BlockfileWriterOptions::new();
        bf_options = bf_options.unordered_mutations();
        match blockfile_provider
            .write::<u32, &SpannPostingList<'_>>(bf_options)
            .await
        {
            Ok(writer) => Ok(writer),
            Err(_) => Err(SpannIndexWriterConstructionError::BlockfileWriterConstructionError),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn from_id(
        hnsw_provider: &HnswIndexProvider,
        hnsw_id: Option<&IndexUuid>,
        versions_map_id: Option<&Uuid>,
        posting_list_id: Option<&Uuid>,
        max_head_id_bf_id: Option<&Uuid>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        collection_id: &CollectionUuid,
        distance_function: DistanceFunction,
        dimensionality: usize,
        blockfile_provider: &BlockfileProvider,
    ) -> Result<Self, SpannIndexWriterConstructionError> {
        // Create the HNSW index.
        let hnsw_index = match hnsw_id {
            Some(hnsw_id) => {
                Self::hnsw_index_from_id(
                    hnsw_provider,
                    hnsw_id,
                    collection_id,
                    distance_function.clone(),
                    dimensionality,
                )
                .await?
            }
            None => {
                Self::create_hnsw_index(
                    hnsw_provider,
                    collection_id,
                    distance_function.clone(),
                    dimensionality,
                    m.unwrap(), // Safe since caller should always provide this.
                    ef_construction.unwrap(), // Safe since caller should always provide this.
                    ef_search.unwrap(), // Safe since caller should always provide this.
                )
                .await?
            }
        };
        // Load the versions map.
        let versions_map = match versions_map_id {
            Some(versions_map_id) => {
                Self::load_versions_map(versions_map_id, blockfile_provider).await?
            }
            None => VersionsMapInner {
                versions_map: HashMap::new(),
            },
        };
        // Fork the posting list writer.
        let posting_list_writer = match posting_list_id {
            Some(posting_list_id) => {
                Self::fork_postings_list(posting_list_id, blockfile_provider).await?
            }
            None => Self::create_posting_list(blockfile_provider).await?,
        };

        let max_head_id = match max_head_id_bf_id {
            Some(max_head_id_bf_id) => {
                let reader = blockfile_provider
                    .read::<&str, u32>(max_head_id_bf_id)
                    .await;
                match reader {
                    Ok(reader) => reader
                        .get("", MAX_HEAD_OFFSET_ID)
                        .await
                        .map_err(|_| {
                            SpannIndexWriterConstructionError::BlockfileReaderConstructionError
                        })?
                        .unwrap(),
                    Err(_) => 1,
                }
            }
            None => 1,
        };
        Ok(Self::new(
            hnsw_index,
            hnsw_provider.clone(),
            blockfile_provider.clone(),
            posting_list_writer,
            max_head_id,
            versions_map,
            distance_function,
            dimensionality,
        ))
    }

    fn add_versions_map(&self, id: u32) -> u32 {
        // 0 means deleted. Version counting starts from 1.
        let mut write_lock = self.versions_map.write();
        write_lock.versions_map.insert(id, 1);
        *write_lock.versions_map.get(&id).unwrap()
    }

    #[allow(dead_code)]
    async fn rng_query(
        &self,
        query: &[f32],
    ) -> Result<(Vec<usize>, Vec<f32>, Vec<Vec<f32>>), SpannIndexWriterConstructionError> {
        let mut normalized_query = query.to_vec();
        // Normalize the query in case of cosine.
        if self.distance_function == DistanceFunction::Cosine {
            normalized_query = normalize(query)
        }
        let ids;
        let distances;
        let mut embeddings: Vec<Vec<f32>> = vec![];
        {
            let read_guard = self.hnsw_index.inner.read();
            let allowed_ids = vec![];
            let disallowed_ids = vec![];
            (ids, distances) = read_guard
                .query(
                    &normalized_query,
                    NUM_CENTROIDS_TO_SEARCH as usize,
                    &allowed_ids,
                    &disallowed_ids,
                )
                .map_err(|_| SpannIndexWriterConstructionError::HnswIndexSearchError)?;
            // Get the embeddings also for distance computation.
            for id in ids.iter() {
                let emb = read_guard
                    .get(*id)
                    .map_err(|_| SpannIndexWriterConstructionError::HnswIndexSearchError)?
                    .ok_or(SpannIndexWriterConstructionError::HnswIndexSearchError)?;
                embeddings.push(emb);
            }
        }
        // Apply the RNG rule to prune.
        let mut res_ids = vec![];
        let mut res_distances = vec![];
        let mut res_embeddings: Vec<Vec<f32>> = vec![];
        // Embeddings that were obtained are already normalized.
        for (id, (distance, embedding)) in ids.iter().zip(distances.iter().zip(embeddings)) {
            let mut rng_accepted = true;
            for nbr_embedding in res_embeddings.iter() {
                let dist = self
                    .distance_function
                    .distance(&embedding[..], &nbr_embedding[..]);
                if RNG_FACTOR * dist <= *distance {
                    rng_accepted = false;
                    break;
                }
            }
            if !rng_accepted {
                continue;
            }
            res_ids.push(*id);
            res_distances.push(*distance);
            res_embeddings.push(embedding);
        }

        Ok((res_ids, res_distances, res_embeddings))
    }

    #[allow(dead_code)]
    async fn append(
        &self,
        head_id: u32,
        id: u32,
        version: u32,
        embedding: &[f32],
        head_embedding: Vec<f32>,
    ) -> Result<(), SpannIndexWriterConstructionError> {
        {
            let write_guard = self.posting_list_writer.lock().await;
            // TODO(Sanket): Check if head is deleted, can happen if another concurrent thread
            // deletes it.
            let (mut doc_offset_ids, mut doc_versions, mut doc_embeddings) = write_guard
                .get_owned::<u32, &SpannPostingList<'_>>("", head_id)
                .await
                .map_err(|_| SpannIndexWriterConstructionError::PostingListSearchError)?
                .ok_or(SpannIndexWriterConstructionError::PostingListSearchError)?;
            // Append the new point to the posting list.
            doc_offset_ids.reserve_exact(1);
            doc_versions.reserve_exact(1);
            doc_embeddings.reserve_exact(embedding.len());
            doc_offset_ids.push(id);
            doc_versions.push(version);
            doc_embeddings.extend_from_slice(embedding);
            // Cleanup this posting list.
            // Note: There is an order in which we are acquiring locks here to prevent deadlocks.
            // Note: This way of cleaning up takes less memory since we don't allocate
            // memory for embeddings that are not outdated.
            let mut local_indices = vec![0; doc_offset_ids.len()];
            let mut up_to_date_index = 0;
            {
                let version_map_guard = self.versions_map.read();
                for (index, doc_version) in doc_versions.iter().enumerate() {
                    let current_version = version_map_guard
                        .versions_map
                        .get(&doc_offset_ids[index])
                        .ok_or(SpannIndexWriterConstructionError::ExpectedDataNotFound)?;
                    // disregard if either deleted or on an older version.
                    if *current_version == 0 || doc_version < current_version {
                        continue;
                    }
                    local_indices[up_to_date_index] = index;
                    up_to_date_index += 1;
                }
            }
            // If size is within threshold, write the new posting back and return.
            if up_to_date_index <= SPLIT_THRESHOLD {
                for idx in 0..up_to_date_index {
                    if local_indices[idx] == idx {
                        continue;
                    }
                    doc_offset_ids[idx] = doc_offset_ids[local_indices[idx]];
                    doc_versions[idx] = doc_versions[local_indices[idx]];
                    doc_embeddings.copy_within(
                        local_indices[idx] * self.dimensionality
                            ..(local_indices[idx] + 1) * self.dimensionality,
                        idx * self.dimensionality,
                    );
                }
                doc_offset_ids.truncate(up_to_date_index);
                doc_versions.truncate(up_to_date_index);
                doc_embeddings.truncate(up_to_date_index * self.dimensionality);
                let posting_list = SpannPostingList {
                    doc_offset_ids: &doc_offset_ids,
                    doc_versions: &doc_versions,
                    doc_embeddings: &doc_embeddings,
                };
                write_guard
                    .set("", head_id, &posting_list)
                    .await
                    .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?;

                return Ok(());
            }
            // Otherwise split the posting list.
            local_indices.truncate(up_to_date_index);
            // Shuffle local_indices.
            local_indices.shuffle(&mut rand::thread_rng());
            let last = local_indices.len();
            // Prepare KMeans.
            let mut kmeans_algo = KMeansAlgorithm::new(
                local_indices,
                &doc_embeddings,
                self.dimensionality,
                /* k */ 2,
                /* first */ 0,
                last,
                NUM_SAMPLES_FOR_KMEANS,
                self.distance_function.clone(),
                INITIAL_LAMBDA,
            );
            let clustering_output = kmeans_algo.cluster();
            // TODO(Sanket): Not sure how this can happen.
            if clustering_output.num_clusters <= 1 {
                tracing::warn!("Clustering split the posting list into only 1 cluster");
                panic!("Clustering split the posting list into only 1 cluster");
            } else {
                let mut new_posting_lists: Vec<Vec<f32>> = Vec::with_capacity(2);
                new_posting_lists[0]
                    .reserve_exact(clustering_output.cluster_counts[0] * self.dimensionality);
                new_posting_lists[1]
                    .reserve_exact(clustering_output.cluster_counts[1] * self.dimensionality);
                let mut new_doc_offset_ids: Vec<Vec<u32>> = Vec::with_capacity(2);
                new_doc_offset_ids[0].reserve_exact(clustering_output.cluster_counts[0]);
                new_doc_offset_ids[1].reserve_exact(clustering_output.cluster_counts[1]);
                let mut new_doc_versions: Vec<Vec<u32>> = Vec::with_capacity(2);
                new_doc_versions[0].reserve_exact(clustering_output.cluster_counts[0]);
                new_doc_versions[1].reserve_exact(clustering_output.cluster_counts[1]);
                for (index, cluster) in clustering_output.cluster_labels {
                    new_doc_offset_ids[cluster as usize].push(doc_offset_ids[index]);
                    new_doc_versions[cluster as usize].push(doc_versions[index]);
                    new_posting_lists[cluster as usize].extend_from_slice(
                        &doc_embeddings
                            [index * self.dimensionality..(index + 1) * self.dimensionality],
                    );
                }
                let mut same_head = false;
                let mut new_head_ids = vec![-1; 2];
                for k in 0..2 {
                    // Update the existing head.
                    if !same_head
                        && self
                            .distance_function
                            .distance(&clustering_output.cluster_centers[k], &head_embedding)
                            < 1e-6
                    {
                        tracing::info!("Same head after splitting");
                        same_head = true;
                        let posting_list = SpannPostingList {
                            doc_offset_ids: &new_doc_offset_ids[k],
                            doc_versions: &new_doc_versions[k],
                            doc_embeddings: &new_posting_lists[k],
                        };
                        write_guard
                            .set("", head_id, &posting_list)
                            .await
                            .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?;
                        new_head_ids[k] = head_id as i32;
                    } else {
                        // Create new head.
                        let next_id = self
                            .next_head_id
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        let posting_list = SpannPostingList {
                            doc_offset_ids: &new_doc_offset_ids[k],
                            doc_versions: &new_doc_versions[k],
                            doc_embeddings: &new_posting_lists[k],
                        };
                        // Insert to postings list.
                        write_guard
                            .set("", next_id, &posting_list)
                            .await
                            .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?;
                        new_head_ids[k] = next_id as i32;
                        // Insert to hnsw now.
                        // TODO(Sanket): Check for capacity and increase as needed.
                        let hnsw_write_guard = self.hnsw_index.inner.write();
                        hnsw_write_guard
                            .add(next_id as usize, &clustering_output.cluster_centers[k])
                            .map_err(|_| SpannIndexWriterConstructionError::HnswIndexAddError)?;
                    }
                }
                if !same_head {
                    // Delete the old head
                    let hnsw_write_guard = self.hnsw_index.inner.write();
                    hnsw_write_guard
                        .delete(head_id as usize)
                        .map_err(|_| SpannIndexWriterConstructionError::HnswIndexAddError)?;
                }
            }
        }
        // Reassign code.
        
        Ok(())
    }

    #[allow(dead_code)]
    async fn add_postings_list(
        &self,
        id: u32,
        version: u32,
        embeddings: &[f32],
    ) -> Result<(), SpannIndexWriterConstructionError> {
        let (ids, _, head_embeddings) = self.rng_query(embeddings).await?;
        // Create a centroid with just this point.
        if ids.is_empty() {
            let next_id = self
                .next_head_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // First add to postings list then to hnsw. This order is important
            // to ensure that if and when the center is discoverable, it also exists
            // in the postings list. Otherwise, it will be a dangling center.
            {
                let posting_list = SpannPostingList {
                    doc_offset_ids: &[id],
                    doc_versions: &[version],
                    doc_embeddings: embeddings,
                };
                let write_guard = self.posting_list_writer.lock().await;
                write_guard
                    .set("", next_id, &posting_list)
                    .await
                    .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?;
            }
            // Next add to hnsw.
            // This shouldn't exceed the capacity since this will happen only for the first few points
            // so no need to check and increase the capacity.
            {
                let write_guard = self.hnsw_index.inner.write();
                write_guard
                    .add(next_id as usize, embeddings)
                    .map_err(|_| SpannIndexWriterConstructionError::HnswIndexAddError)?;
            }
            return Ok(());
        }
        // Otherwise add to the posting list of these arrays.
        for (head_id, head_embedding) in ids.iter().zip(head_embeddings) {
            self.append(*head_id as u32, id, version, embeddings, head_embedding)
                .await?;
        }

        Ok(())
    }

    pub async fn add(
        &self,
        id: u32,
        embedding: &[f32],
    ) -> Result<(), SpannIndexWriterConstructionError> {
        let version = self.add_versions_map(id);
        // Add to the posting list.
        self.add_postings_list(id, version, embedding).await
    }

    // TODO(Sanket): Change the error types.
    pub async fn commit(self) -> Result<SpannIndexFlusher, SpannIndexWriterConstructionError> {
        // Pl list.
        let pl_flusher = match Arc::try_unwrap(self.posting_list_writer) {
            Ok(writer) => writer
                .into_inner()
                .commit::<u32, &SpannPostingList<'_>>()
                .await
                .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?,
            Err(_) => {
                // This should never happen.
                panic!("Failed to unwrap posting list writer");
            }
        };
        // Versions map. Create a writer, write all the data and commit.
        let mut bf_options = BlockfileWriterOptions::new();
        bf_options = bf_options.unordered_mutations();
        let versions_map_bf_writer = self
            .blockfile_provider
            .write::<u32, u32>(bf_options)
            .await
            .map_err(|_| SpannIndexWriterConstructionError::BlockfileWriterConstructionError)?;
        let versions_map_flusher = match Arc::try_unwrap(self.versions_map) {
            Ok(writer) => {
                let writer = writer.into_inner();
                for (doc_offset_id, doc_version) in writer.versions_map.into_iter() {
                    versions_map_bf_writer
                        .set("", doc_offset_id, doc_version)
                        .await
                        .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?;
                }
                versions_map_bf_writer
                    .commit::<u32, u32>()
                    .await
                    .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?
            }
            Err(_) => {
                // This should never happen.
                panic!("Failed to unwrap posting list writer");
            }
        };
        // Next head.
        let mut bf_options = BlockfileWriterOptions::new();
        bf_options = bf_options.unordered_mutations();
        let max_head_id_bf = self
            .blockfile_provider
            .write::<&str, u32>(bf_options)
            .await
            .map_err(|_| SpannIndexWriterConstructionError::BlockfileWriterConstructionError)?;
        let max_head_id_flusher = match Arc::try_unwrap(self.next_head_id) {
            Ok(value) => {
                let value = value.into_inner();
                max_head_id_bf
                    .set("", MAX_HEAD_OFFSET_ID, value)
                    .await
                    .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?;
                max_head_id_bf
                    .commit::<&str, u32>()
                    .await
                    .map_err(|_| SpannIndexWriterConstructionError::PostingListAddError)?
            }
            Err(_) => {
                // This should never happen.
                panic!("Failed to unwrap next head id");
            }
        };

        let hnsw_id = self.hnsw_index.inner.read().id;

        // Hnsw.
        self.hnsw_provider
            .commit(self.hnsw_index)
            .map_err(|_| SpannIndexWriterConstructionError::HnswIndexConstructionError)?;

        Ok(SpannIndexFlusher {
            pl_flusher,
            versions_map_flusher,
            max_head_id_flusher,
            hnsw_id,
            hnsw_flusher: self.hnsw_provider,
        })
    }
}

pub struct SpannIndexFlusher {
    pl_flusher: BlockfileFlusher,
    versions_map_flusher: BlockfileFlusher,
    max_head_id_flusher: BlockfileFlusher,
    hnsw_id: IndexUuid,
    hnsw_flusher: HnswIndexProvider,
}

pub struct SpannIndexIds {
    pub pl_id: Uuid,
    pub versions_map_id: Uuid,
    pub max_head_id_id: Uuid,
    pub hnsw_id: IndexUuid,
}

// TODO(Sanket): Change the error types.
impl SpannIndexFlusher {
    pub async fn flush(self) -> Result<SpannIndexIds, SpannIndexWriterConstructionError> {
        let res = SpannIndexIds {
            pl_id: self.pl_flusher.id(),
            versions_map_id: self.versions_map_flusher.id(),
            max_head_id_id: self.max_head_id_flusher.id(),
            hnsw_id: self.hnsw_id,
        };
        self.pl_flusher
            .flush::<u32, &SpannPostingList<'_>>()
            .await
            .map_err(|_| SpannIndexWriterConstructionError::BlockfileWriterConstructionError)?;
        self.versions_map_flusher
            .flush::<u32, u32>()
            .await
            .map_err(|_| SpannIndexWriterConstructionError::BlockfileWriterConstructionError)?;
        self.max_head_id_flusher
            .flush::<&str, u32>()
            .await
            .map_err(|_| SpannIndexWriterConstructionError::BlockfileWriterConstructionError)?;
        self.hnsw_flusher
            .flush(&self.hnsw_id)
            .await
            .map_err(|_| SpannIndexWriterConstructionError::HnswIndexConstructionError)?;
        Ok(res)
    }
}
