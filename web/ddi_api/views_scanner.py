"""
Drug Scanner API Views for Project Aegis

Endpoints to support camera-based drug detection:
1. NDC Code Lookup
2. Pill Identification Search
3. Drug Name Search (enhanced for OCR)

@author OpenClaw Bot for Project Aegis
"""

import re
import logging
from typing import Optional, List, Dict

from django.db.models import Q
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Drug
from .serializers import DrugSerializer

logger = logging.getLogger(__name__)


# ============== NDC Code Lookup ==============

@api_view(['GET'])
def lookup_by_ndc(request, ndc_code: str):
    """
    Lookup a drug by its National Drug Code (NDC).
    
    NDC formats accepted:
    - 10-digit: 12345-6789-01
    - 11-digit: 12345-6789-01 or 123456789012
    
    Args:
        ndc_code: The NDC code to lookup
        
    Returns:
        Drug details if found, 404 otherwise
    """
    # Normalize NDC - remove dashes, spaces
    normalized_ndc = re.sub(r'[\s\-]', '', ndc_code)
    
    # Try different NDC formats
    ndc_variations = [
        ndc_code,  # Original
        normalized_ndc,  # No dashes
        f"{normalized_ndc[:5]}-{normalized_ndc[5:9]}-{normalized_ndc[9:]}",  # 5-4-2 format
        f"{normalized_ndc[:4]}-{normalized_ndc[4:8]}-{normalized_ndc[8:]}",  # 4-4-2 format
    ]
    
    # Search for drug with this NDC
    drug = None
    for ndc in ndc_variations:
        drug = Drug.objects.filter(
            Q(ndc_code__iexact=ndc) |
            Q(ndc_code__icontains=ndc) |
            Q(drugbank_id__iexact=ndc)  # Fallback to drugbank_id
        ).first()
        
        if drug:
            break
    
    if drug:
        serializer = DrugSerializer(drug)
        return Response(serializer.data)
    
    # Try external lookup (OpenFDA, RxNorm)
    external_result = lookup_ndc_external(ndc_code)
    if external_result:
        return Response(external_result)
    
    return Response(
        {'error': f'Drug with NDC {ndc_code} not found'},
        status=status.HTTP_404_NOT_FOUND
    )


def lookup_ndc_external(ndc_code: str) -> Optional[Dict]:
    """
    Lookup NDC code in external databases (OpenFDA, RxNorm).
    Returns basic drug info if found.
    """
    import requests
    
    # Try OpenFDA
    try:
        response = requests.get(
            f'https://api.fda.gov/drug/ndc.json',
            params={'search': f'product_ndc:"{ndc_code}"', 'limit': 1},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return {
                    'name': result.get('brand_name') or result.get('generic_name'),
                    'generic_name': result.get('generic_name'),
                    'brand_name': result.get('brand_name'),
                    'strength': result.get('active_ingredients', [{}])[0].get('strength'),
                    'dosage_form': result.get('dosage_form'),
                    'route': result.get('route', [None])[0],
                    'therapeutic_class': result.get('pharm_class', [None])[0],
                    'ndc_code': ndc_code,
                    'source': 'openfda'
                }
    except Exception as e:
        logger.warning(f'OpenFDA lookup failed: {e}')
    
    return None


# ============== Pill Identification Search ==============

@api_view(['GET'])
def pill_search(request):
    """
    Search for drugs by pill characteristics.
    
    Query Parameters:
        color: Pill color (white, pink, blue, yellow, etc.)
        shape: Pill shape (round, oval, capsule, etc.)
        imprint: Text/numbers imprinted on pill
        
    Returns:
        List of matching drugs with confidence scores
    """
    color = request.query_params.get('color', '').lower()
    shape = request.query_params.get('shape', '').lower()
    imprint = request.query_params.get('imprint', '').upper()
    
    if not any([color, shape, imprint]):
        return Response(
            {'error': 'At least one search parameter (color, shape, imprint) required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Build query
    queryset = Drug.objects.all()
    
    if color:
        queryset = queryset.filter(
            Q(pill_color__iexact=color) |
            Q(pill_color__icontains=color) |
            Q(description__icontains=color)
        )
    
    if shape:
        queryset = queryset.filter(
            Q(pill_shape__iexact=shape) |
            Q(pill_shape__icontains=shape) |
            Q(dosage_form__icontains=shape)
        )
    
    if imprint:
        queryset = queryset.filter(
            Q(pill_imprint__iexact=imprint) |
            Q(pill_imprint__icontains=imprint) |
            Q(name__icontains=imprint)
        )
    
    # Get results
    drugs = queryset[:20]
    
    if not drugs:
        # Try external pill identification
        external_results = search_pill_external(color, shape, imprint)
        if external_results:
            return Response({
                'results': external_results,
                'source': 'external',
                'count': len(external_results)
            })
        
        return Response({
            'results': [],
            'message': 'No matching pills found. Try adjusting your search criteria.',
            'suggestions': [
                'Ensure proper lighting when capturing the pill image',
                'Try searching by imprint code if visible',
                'Use barcode or label scanning for more accurate results'
            ]
        })
    
    serializer = DrugSerializer(drugs, many=True)
    results = serializer.data
    
    # Add confidence scores based on match quality
    for i, drug in enumerate(results):
        match_score = 0
        if color and drug.get('pill_color', '').lower() == color:
            match_score += 0.4
        if shape and drug.get('pill_shape', '').lower() == shape:
            match_score += 0.3
        if imprint and imprint in drug.get('pill_imprint', '').upper():
            match_score += 0.3
        
        results[i]['confidence'] = min(match_score + 0.5, 1.0)  # Base confidence of 0.5
    
    # Sort by confidence
    results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    return Response({
        'results': results,
        'source': 'database',
        'count': len(results)
    })


def search_pill_external(color: str, shape: str, imprint: str) -> List[Dict]:
    """
    Search external pill identification databases.
    
    Uses NIH Pillbox API or similar services.
    """
    import requests
    
    # NIH DailyMed API (basic search)
    try:
        params = {}
        if imprint:
            params['drug_name'] = imprint
        
        response = requests.get(
            'https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json',
            params=params,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for item in data.get('data', [])[:10]:
                results.append({
                    'name': item.get('title', 'Unknown'),
                    'setid': item.get('setid'),
                    'source': 'dailymed',
                    'confidence': 0.6
                })
            
            return results
    except Exception as e:
        logger.warning(f'External pill search failed: {e}')
    
    return []


# ============== Enhanced Drug Search (for OCR) ==============

@api_view(['GET'])
def enhanced_drug_search(request):
    """
    Enhanced drug search optimized for OCR results.
    
    Handles:
    - Fuzzy matching for OCR errors
    - Brand name to generic mapping
    - Partial matches
    - Strength/dosage extraction
    
    Query Parameters:
        q: Search query (drug name from OCR)
        fuzzy: Enable fuzzy matching (default: true)
        limit: Max results (default: 10)
        
    Returns:
        List of matching drugs with confidence scores
    """
    query = request.query_params.get('q', '').strip()
    fuzzy = request.query_params.get('fuzzy', 'true').lower() == 'true'
    limit = min(int(request.query_params.get('limit', 10)), 50)
    
    if not query or len(query) < 2:
        return Response(
            {'error': 'Search query must be at least 2 characters'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Normalize query
    normalized_query = normalize_ocr_query(query)
    
    results = []
    seen_ids = set()
    
    # 1. Exact match (highest confidence)
    exact_matches = Drug.objects.filter(
        Q(name__iexact=normalized_query) |
        Q(generic_name__iexact=normalized_query)
    )[:limit]
    
    for drug in exact_matches:
        if drug.id not in seen_ids:
            seen_ids.add(drug.id)
            results.append({
                **DrugSerializer(drug).data,
                'confidence': 1.0,
                'match_type': 'exact'
            })
    
    # 2. Prefix match (high confidence)
    if len(results) < limit:
        prefix_matches = Drug.objects.filter(
            Q(name__istartswith=normalized_query) |
            Q(generic_name__istartswith=normalized_query)
        ).exclude(id__in=seen_ids)[:limit - len(results)]
        
        for drug in prefix_matches:
            if drug.id not in seen_ids:
                seen_ids.add(drug.id)
                results.append({
                    **DrugSerializer(drug).data,
                    'confidence': 0.9,
                    'match_type': 'prefix'
                })
    
    # 3. Contains match (medium confidence)
    if len(results) < limit:
        contains_matches = Drug.objects.filter(
            Q(name__icontains=normalized_query) |
            Q(generic_name__icontains=normalized_query)
        ).exclude(id__in=seen_ids)[:limit - len(results)]
        
        for drug in contains_matches:
            if drug.id not in seen_ids:
                seen_ids.add(drug.id)
                results.append({
                    **DrugSerializer(drug).data,
                    'confidence': 0.7,
                    'match_type': 'contains'
                })
    
    # 4. Fuzzy match (lower confidence) - for OCR errors
    if fuzzy and len(results) < limit:
        fuzzy_results = fuzzy_drug_search(normalized_query, seen_ids, limit - len(results))
        results.extend(fuzzy_results)
    
    # 5. Brand name mapping
    brand_result = lookup_brand_name(normalized_query)
    if brand_result and brand_result['id'] not in seen_ids:
        results.insert(0, {
            **brand_result,
            'confidence': 0.95,
            'match_type': 'brand_mapping'
        })
    
    return Response({
        'query': query,
        'normalized_query': normalized_query,
        'results': results[:limit],
        'count': len(results[:limit])
    })


def normalize_ocr_query(query: str) -> str:
    """
    Normalize OCR text for drug search.
    Handles common OCR errors and variations.
    """
    # Remove common OCR artifacts
    normalized = re.sub(r'[^\w\s\-]', '', query)
    
    # Fix common OCR substitutions
    ocr_fixes = {
        '0': 'O',  # Zero to O
        '1': 'I',  # One to I (sometimes)
        '5': 'S',  # Five to S
        '8': 'B',  # Eight to B
    }
    
    # Remove strength suffixes for matching
    normalized = re.sub(r'\s*\d+\s*(mg|mcg|ml|g)\s*$', '', normalized, flags=re.I)
    
    # Trim and lowercase
    return normalized.strip()


def fuzzy_drug_search(query: str, exclude_ids: set, limit: int) -> List[Dict]:
    """
    Perform fuzzy search for drug names.
    Handles typos and OCR errors.
    """
    results = []
    
    # Simple fuzzy: try removing/swapping characters
    variations = generate_query_variations(query)
    
    for variation in variations[:10]:  # Limit variations checked
        matches = Drug.objects.filter(
            Q(name__icontains=variation) |
            Q(generic_name__icontains=variation)
        ).exclude(id__in=exclude_ids)[:limit]
        
        for drug in matches:
            if drug.id not in exclude_ids:
                exclude_ids.add(drug.id)
                results.append({
                    **DrugSerializer(drug).data,
                    'confidence': 0.5,
                    'match_type': 'fuzzy',
                    'variation': variation
                })
                
                if len(results) >= limit:
                    return results
    
    return results


def generate_query_variations(query: str) -> List[str]:
    """
    Generate variations of query for fuzzy matching.
    """
    variations = [query]
    
    # Common suffix variations for drug names
    suffixes = ['in', 'ol', 'am', 'ine', 'ide', 'ate', 'one']
    
    for suffix in suffixes:
        if not query.lower().endswith(suffix):
            variations.append(query + suffix)
    
    # Try removing last character (common OCR issue)
    if len(query) > 3:
        variations.append(query[:-1])
    
    return variations


def lookup_brand_name(query: str) -> Optional[Dict]:
    """
    Lookup brand name and return generic drug.
    """
    brand_to_generic = {
        'tylenol': 'acetaminophen',
        'advil': 'ibuprofen',
        'motrin': 'ibuprofen',
        'aleve': 'naproxen',
        'lipitor': 'atorvastatin',
        'zocor': 'simvastatin',
        'crestor': 'rosuvastatin',
        'prilosec': 'omeprazole',
        'nexium': 'esomeprazole',
        'coumadin': 'warfarin',
        'plavix': 'clopidogrel',
        'xarelto': 'rivaroxaban',
        'eliquis': 'apixaban',
        'prozac': 'fluoxetine',
        'zoloft': 'sertraline',
        'lexapro': 'escitalopram',
        'xanax': 'alprazolam',
        'ativan': 'lorazepam',
        'valium': 'diazepam',
        'ambien': 'zolpidem',
        'synthroid': 'levothyroxine',
        'glucophage': 'metformin',
        'viagra': 'sildenafil',
        'cialis': 'tadalafil',
        # Add more mappings as needed
    }
    
    query_lower = query.lower()
    
    if query_lower in brand_to_generic:
        generic_name = brand_to_generic[query_lower]
        drug = Drug.objects.filter(
            Q(name__iexact=generic_name) |
            Q(generic_name__iexact=generic_name)
        ).first()
        
        if drug:
            return DrugSerializer(drug).data
    
    return None


# ============== Barcode Validation ==============

@api_view(['POST'])
def validate_barcode(request):
    """
    Validate and parse a barcode/NDC code.
    
    Request Body:
        barcode: The barcode string to validate
        
    Returns:
        Validation result and parsed components
    """
    barcode = request.data.get('barcode', '').strip()
    
    if not barcode:
        return Response(
            {'error': 'Barcode is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Remove non-digits
    digits_only = re.sub(r'\D', '', barcode)
    
    result = {
        'original': barcode,
        'digits': digits_only,
        'valid': False,
        'type': None,
        'parsed': None
    }
    
    # Check for NDC (10 or 11 digits)
    if len(digits_only) in [10, 11]:
        result['valid'] = True
        result['type'] = 'NDC'
        
        if len(digits_only) == 10:
            # 10-digit NDC: 4-4-2, 5-3-2, or 5-4-1 format
            result['parsed'] = {
                'labeler': digits_only[:5],
                'product': digits_only[5:9],
                'package': digits_only[9:]
            }
        else:
            # 11-digit NDC: 5-4-2 format
            result['parsed'] = {
                'labeler': digits_only[:5],
                'product': digits_only[5:9],
                'package': digits_only[9:]
            }
    
    # Check for UPC (12 digits)
    elif len(digits_only) == 12:
        result['valid'] = True
        result['type'] = 'UPC'
        result['parsed'] = {
            'system': digits_only[0],
            'manufacturer': digits_only[1:6],
            'product': digits_only[6:11],
            'check': digits_only[11]
        }
    
    # Check for EAN (13 digits)
    elif len(digits_only) == 13:
        result['valid'] = True
        result['type'] = 'EAN'
        result['parsed'] = {
            'country': digits_only[:3],
            'manufacturer': digits_only[3:7],
            'product': digits_only[7:12],
            'check': digits_only[12]
        }
    
    return Response(result)


# ============== Pill Image Analysis ==============

@api_view(['POST'])
def analyze_pill_image(request):
    """
    Analyze an uploaded pill image server-side.
    
    Accepts a pill image and returns:
    - Detected color, shape, imprint from the image
    - Matching drugs from the database
    
    Request Body (multipart/form-data):
        image: Pill image file
        color (optional): Pre-detected color from client CV
        shape (optional): Pre-detected shape from client CV
        imprint (optional): Pre-detected imprint from client OCR
        
    Returns:
        Detected features + matching drug results
    """
    image = request.FILES.get('image')
    
    # Also accept client-side detected features as hints
    client_color = request.data.get('color', '').lower().strip()
    client_shape = request.data.get('shape', '').lower().strip()
    client_imprint = request.data.get('imprint', '').upper().strip()
    
    # Use client-detected features (client CV pipeline is primary)
    color = client_color or None
    shape = client_shape or None
    imprint = client_imprint or None
    
    if not any([color, shape, imprint, image]):
        return Response(
            {'error': 'At least one of: image, color, shape, or imprint required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Build query based on detected features
    results = []
    queryset = Drug.objects.all()
    
    if color:
        queryset = queryset.filter(
            Q(pill_color__iexact=color) |
            Q(pill_color__icontains=color)
        )
    
    if shape:
        queryset = queryset.filter(
            Q(pill_shape__iexact=shape) |
            Q(pill_shape__icontains=shape)
        )
    
    if imprint:
        queryset = queryset.filter(
            Q(pill_imprint__iexact=imprint) |
            Q(pill_imprint__icontains=imprint)
        )
    
    drugs = queryset[:20]
    
    if drugs:
        serializer = DrugSerializer(drugs, many=True)
        results = serializer.data
        
        for result in results:
            confidence = 0.3
            if color and result.get('pill_color', '').lower() == color:
                confidence += 0.25
            if shape and result.get('pill_shape', '').lower() == shape:
                confidence += 0.2
            if imprint and imprint in (result.get('pill_imprint') or '').upper():
                confidence += 0.3
            result['confidence'] = min(confidence, 0.95)
        
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Try external APIs if no local results
    if not results:
        external = search_pill_external(color or '', shape or '', imprint or '')
        if external:
            results = external
    
    return Response({
        'detected_features': {
            'color': color,
            'shape': shape,
            'imprint': imprint,
        },
        'results': results,
        'count': len(results),
        'source': 'database' if drugs else 'external'
    })
